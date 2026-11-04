"""Causal metric cuboid fitting from calibrated, complete image silhouettes.

The box is a generic elongated vehicle volume, not a supplied mesh. Its scale is
learned only from simultaneous views, and its front sign only from past/current
motion (forward travel is an explicit assumption). A known-size single view can
measure range; it cannot independently re-measure the learned physical size.
No reference trajectory, image from a later frame, or fixed pivot offset is used.

Axis-aligned detector boxes are imperfect silhouettes. Fits carry a model-error
floor, reject collapsed volumes/poor geometry, and never train on clipped boxes.
Front-face center is a geometric convention, not a guarantee about a mesh pivot.
"""
from collections import deque
from itertools import combinations, product
import math

import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize_scalar


_SIGNS = np.array(list(product((-1., 1.), repeat=3)))


def _axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _vector(value, name):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain three finite coordinates')
    return result


class ShapeEstimator:
    """Shared-size, upright cuboid estimator. Inputs/outputs use UE centimetres.

    ``position_cm`` always names the volume center. A caller selecting ``front_cm``
    must also select ``front_covariance_cm2`` and expose that reference convention.
    ``front_cm`` is None while the 180-degree orientation ambiguity is unresolved.
    ``metric_depth_observed`` means simultaneous calibrated views; single-view
    range is separately exposed by ``range_from_shape`` and ``metric_scale_source``.

    C (``covariance_cm2``) is the conservative per-observation marginal used by
    the existing filter's gain/gate. S (``systematic_covariance_cm2``) is an
    additional persistent common-error floor for reporting after repeated
    updates. C already contains some model floors: C/S are deliberately not an
    independent-noise decomposition or a calibrated confidence interval.
    """

    def __init__(self, rig):
        self.rig = rig
        self.reset()

    def reset(self):
        self._frame = None
        self._timestamp = None
        self._samples = deque(maxlen=40)
        self._sample_count = 0
        self._dimensions = None
        self._dimension_variance = None
        self._centers = deque(maxlen=90)
        self._yaw = None
        self._heading = None
        self._last_full_yaw = None
        self._last_full_timestamp = None

    def _views(self, frame, observations):
        selected = {}
        margin = max(2., .002 * self.rig.width)
        for observation in observations:
            try:
                if observation.get('crop_only', False) or observation.get('measured') is False:
                    continue
                quality = observation.get('quality') or {}
                fill = float(quality.get('foreground_fill', 1.))
                if not math.isfinite(fill) or not 0 < fill <= 1:
                    continue
                camera_id = int(observation['camera_id'])
                box = np.asarray(observation['bbox'], dtype=float)
                confidence = float(observation['confidence'])
                if (box.shape != (4,) or not np.isfinite(box).all()
                        or not math.isfinite(confidence) or not 0 < confidence <= 1):
                    continue
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= self.rig.width
                        and 0 <= y1 < y2 <= self.rig.height
                        and min(x2-x1, y2-y1) >= 10.):
                    continue
                clipped = np.array([x1 <= margin, x2 >= self.rig.width-margin,
                                    y1 <= margin, y2 >= self.rig.height-margin])
                explicit_partial = bool(observation.get('occluded') or observation.get('truncated')
                                        or quality.get('partial'))
                partial = bool(clipped.any() or explicit_partial or fill < .6)
                if explicit_partial:
                    clipped[:] = True
                camera, origin, rotation = self.rig._pose(camera_id, frame)
                _, direction = self.rig.ray(camera_id, frame, ((x1+x2)/2, (y1+y2)/2))
                local = np.array([
                    [-(x1-self.rig.width/2)/camera.fx, 1., 0.],
                    [(x2-self.rig.width/2)/camera.fx, -1., 0.],
                    [(self.rig.height/2-y1)/camera.fy, 0., -1.],
                    [(y2-self.rig.height/2)/camera.fy, 0., 1.],
                ])
                norms = np.linalg.norm(local, axis=1)
                normals = (local / norms[:, None]) @ rotation.T
                # Extrema order throughout is left, right, top, bottom.
                sigma = np.array([max(2., .0125*(x2-x1))]*2 + [max(2., .02*(y2-y1))]*2)
                if fill < .6:
                    sigma *= 3.
                view = dict(camera_id=camera_id, origin=origin/100., rotation=rotation,
                            direction=direction, box=box, edges=box[[0, 2, 1, 3]],
                            fx=camera.fx, fy=camera.fy, normals=normals, norms=norms,
                            sigma=sigma, confidence=confidence, clipped=clipped,
                            partial=partial, learnable=not partial)
                if camera_id not in selected or confidence > selected[camera_id]['confidence']:
                    selected[camera_id] = view
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return [selected[key] for key in sorted(selected)]

    @staticmethod
    def _seed(views):
        if len(views) < 2:
            return None
        if not any(np.linalg.norm(a['origin']-b['origin']) > 1.
                   and np.linalg.norm(np.cross(a['direction'], b['direction'])) > math.sin(math.radians(8))
                   for a, b in combinations(views, 2)):
            return None
        matrices = [np.eye(3)-np.outer(v['direction'], v['direction']) for v in views]
        matrix = sum(matrices)
        if np.linalg.cond(matrix) > 1e4:
            return None
        center = np.linalg.solve(matrix, sum(m @ v['origin'] for m, v in zip(matrices, views)))
        if any((center-v['origin']) @ v['direction'] <= 0 for v in views):
            return None
        return center

    def _motion_heading(self, timestamp, robust_center, velocity):
        if velocity is not None and np.linalg.norm(velocity[:2]) >= 50.:
            return math.atan2(velocity[1], velocity[0])
        if robust_center is not None:
            for old_time, old_center in self._centers:
                age = timestamp-old_time
                delta = robust_center-old_center
                if .25 <= age <= 2. and np.linalg.norm(delta[:2]) >= 1.:
                    return math.atan2(delta[1], delta[0])
        return None

    def _project(self, center, yaw, dimensions, views):
        corners = center + (_SIGNS*dimensions/2) @ _axes(yaw).T
        result = []
        for view in views:
            local = (corners-view['origin']) @ view['rotation']
            depth = local[:, 0]
            if depth.min() <= .05:
                return None
            u = self.rig.width/2 + view['fx']*local[:, 1]/depth
            v = self.rig.height/2 - view['fy']*local[:, 2]/depth
            result.extend((u.min(), u.max(), v.min(), v.max()))
        return np.asarray(result)

    @staticmethod
    def _plausible(dimensions):
        length, width, height = dimensions
        return (1. < length < 100. and .3 < width < 40. and .3 < height < 30.
                and 1.6 < length/width < 8. and .08 < height/length < .8)

    @staticmethod
    def _parameter_covariance(jacobian):
        """Invert information without turning weak modes into zero variance.

        Forming J.T@J squares its condition number; a pseudoinverse can then
        discard a weak depth mode and falsely report certainty in that mode.
        Missing rank is rejected, while weak nonzero information retains its
        correspondingly large variance through a direct SVD of J.
        """
        jacobian = np.asarray(jacobian, dtype=float)
        if (jacobian.ndim != 2 or jacobian.shape[0] < jacobian.shape[1]
                or not np.isfinite(jacobian).all()):
            return None
        _, singular, basis = np.linalg.svd(jacobian, full_matrices=False)
        if not len(singular) or singular[0] <= 0:
            return None
        tolerance = 16*np.finfo(float).eps*max(jacobian.shape)*singular[0]
        if singular[-1] <= tolerance:
            return None
        covariance = (basis.T / singular**2) @ basis
        return (covariance+covariance.T)/2 if np.isfinite(covariance).all() else None

    def _fit(self, views, seed, heading, shared=None, fixed=False):
        """Frustum tangent planes seed a robust exact-pixel refinement."""
        orientation_held = bool(fixed and all(v['partial'] for v in views)
                                and self._last_full_yaw is not None)
        if orientation_held:
            heading = self._last_full_yaw
        normals = np.vstack([v['normals'] for v in views])
        rhs = np.concatenate([v['normals'] @ v['origin'] for v in views])
        weights = []
        for view in views:
            depth = max(1., (seed-view['origin']) @ view['rotation'][:, 0])
            weights.extend(np.array([view['fx']]*2+[view['fy']]*2)*view['norms']/depth/view['sigma'])
        weights = np.asarray(weights)
        target = np.concatenate([v['edges'] for v in views])
        sigma = np.concatenate([v['sigma'] for v in views])
        clipped = np.concatenate([v['clipped'] for v in views])
        weights[clipped] = 0.
        if np.count_nonzero(~clipped) < (3 if fixed else 6):
            return None
        shared_sigma = np.array([.12, .25, .15])
        if shared is not None and self._dimension_variance is not None:
            shared_sigma = np.maximum(shared_sigma, np.sqrt(self._dimension_variance)/shared)

        def edge_error(predicted):
            errors = predicted-target
            # A clipped object extends beyond an image border. The border gives
            # a one-sided constraint; it is not an observed silhouette extremum.
            signs = np.tile([1., -1., 1., -1.], len(views))
            errors[clipped] = signs[clipped]*np.maximum(0., signs[clipped]*errors[clipped])
            return errors

        def linear(yaw):
            extent = np.abs(normals @ _axes(yaw))
            if fixed:
                a, b = normals*weights[:, None], (rhs+extent@(shared/2))*weights
                center = np.linalg.lstsq(a, b, rcond=None)[0]
                dimensions = shared
            else:
                a = np.column_stack((normals, -extent))*weights[:, None]
                b = rhs*weights
                if shared is not None:
                    prior = np.zeros((3, 6))
                    prior[:, 3:] = np.diag(2./(shared*shared_sigma))
                    a = np.vstack((a, prior))
                    b = np.r_[b, 1./shared_sigma]
                fit = np.linalg.lstsq(a, b, rcond=None)[0]
                if (fit[3:] <= .005).any():
                    fit = lsq_linear(a, b, bounds=([-np.inf]*3+[.005]*3,
                                                  [np.inf]*3+[100.]*3), tol=1e-6).x
                center, dimensions = fit[:3], fit[3:]*2
            predicted = self._project(center, yaw, dimensions, views)
            if predicted is None or not np.isfinite(predicted).all():
                return 1e12, center, dimensions
            error = edge_error(predicted)/sigma
            score = float(error @ error)
            if heading is not None:
                score += (math.sin(yaw-heading)/.18)**2
            if shared is not None and not fixed:
                score += float(np.sum((np.log(np.maximum(dimensions, .001)/shared)/shared_sigma)**2))
            return score, center, dimensions

        # A single-view yaw needs a causal orientation prior. Its alternatives
        # otherwise give different depths while exactly fitting the same box.
        orientation = heading if heading is not None else self._yaw
        if orientation_held:
            grid = np.array([self._last_full_yaw])
        elif fixed and len(views) == 1:
            if orientation is None:
                return None
            grid = orientation + np.linspace(-.55, .55, 9)
        else:
            grid = np.arange(0., math.pi, math.pi/12)
            if orientation is not None:
                grid = np.r_[grid, orientation % math.pi]
        candidates = sorted((linear(float(yaw))[0], float(yaw)) for yaw in grid)
        finalists = []
        for _, yaw in candidates[:3]:
            result = minimize_scalar(lambda angle: linear(angle)[0],
                                     bounds=(yaw-math.pi/12, yaw+math.pi/12),
                                     method='bounded', options={'xatol': 1e-5, 'maxiter': 35})
            finalists.append((linear(result.x)[0], result.x))
        _, yaw = min(finalists + candidates[:1])
        if orientation_held:
            yaw = self._last_full_yaw
        _, center, dimensions = linear(yaw)
        if dimensions[0] < dimensions[1] and not fixed:
            dimensions = dimensions[[1, 0, 2]]
            yaw += math.pi/2
        motion_constrained = False
        if (not fixed and shared is None and heading is not None
                and dimensions[0]/max(dimensions[1], .001) <= 1.6):
            # Wide, rotated alternatives are a known two-view silhouette
            # ambiguity. A motion-supported long axis may resolve it. A collapsed
            # cross section is never rescued this way.
            yaw = heading
            _, center, dimensions = linear(yaw)
            motion_constrained = True
        if not np.isfinite(np.r_[center, dimensions, yaw]).all():
            return None
        initial = np.r_[center, yaw] if fixed else np.r_[center, yaw, dimensions]
        motion_sigma = .002 if orientation_held else .1 if motion_constrained else .18
        orientation = heading if heading is not None else self._yaw if fixed else None

        def residual(parameters):
            dims = shared if fixed else parameters[4:]
            predicted = self._project(parameters[:3], parameters[3], dims, views)
            values = np.full(len(target), 1e5) if predicted is None else edge_error(predicted)/sigma
            if orientation is not None:
                values = np.r_[values, math.sin(parameters[3]-orientation)/motion_sigma]
            if shared is not None and not fixed:
                values = np.r_[values, np.log(dims/shared)/shared_sigma]
            return values

        yaw_span = 1e-6 if orientation_held else .3 if motion_constrained else math.pi/2
        lower = [-np.inf]*3 + [yaw-yaw_span] + ([] if fixed else [.01]*3)
        upper = [np.inf]*3 + [yaw+yaw_span] + ([] if fixed else [200.]*3)
        initial = np.clip(initial, np.asarray(lower)+1e-8, np.asarray(upper)-1e-8)
        result = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1',
                               f_scale=2., x_scale='jac', max_nfev=65,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        parameters = result.x
        center, yaw = parameters[:3], float(parameters[3])
        dimensions = shared.copy() if fixed else parameters[4:].copy()
        if not fixed and dimensions[0] < dimensions[1]:
            # The optimized axis exchange is valid, but the covariance must use
            # the same parameter ordering as the returned canonical long axis.
            order = [0, 1, 2, 3, 5, 4, 6]
            parameters = parameters[order]
            result.jac = result.jac[:, order]
            dimensions = parameters[4:].copy()
            yaw += math.pi/2
        projected = self._project(center, yaw, dimensions, views)
        if projected is None or not self._plausible(dimensions):
            return None
        errors = edge_error(projected)
        normalized = errors/sigma
        if np.sqrt(np.mean(normalized**2)) > 2.5 or np.max(np.abs(normalized)) > 4.:
            return None
        # Orientation/size priors cannot provide an otherwise missing image
        # position constraint, including satisfied containment inequalities.
        if self._parameter_covariance(result.jac[:len(target), :3]) is None:
            return None
        covariance = self._parameter_covariance(result.jac)
        if covariance is None:
            return None
        position_sigma = math.sqrt(max(0., float(np.linalg.eigvalsh(covariance[:3, :3]).max())))
        nearest_range = min(np.linalg.norm(center-view['origin']) for view in views)
        if position_sigma > .5*nearest_range:
            return None
        active_camera_ids = []
        metric_camera_ids = []
        for index, view in enumerate(views):
            image_jacobian = result.jac[index*4:(index+1)*4, :3]
            if np.linalg.matrix_rank(image_jacobian) > 0:
                active_camera_ids.append(view['camera_id'])
            # Satisfied containment constraints contribute no information. For
            # independent-view depth, conservatively require two genuine image
            # edge constraints from that camera, not merely a second camera ID.
            genuine = image_jacobian[~view['clipped']]
            if len(genuine) and np.linalg.matrix_rank(genuine) >= 2:
                metric_camera_ids.append(view['camera_id'])
        # Uncertainty uses detector-edge noise and a model floor, not optimizer
        # residual alone: an exact fit does not establish an exact physical box.
        yaw_sigma = math.sqrt(max(0., covariance[3, 3]))
        if orientation_held:
            age = max(0., self._timestamp-self._last_full_timestamp)
            yaw_sigma = max(yaw_sigma, .18 + min(math.pi/2, age*math.radians(20.)))
            covariance[3, 3] = yaw_sigma**2
        if not fixed:
            relative_sigma = np.sqrt(np.maximum(0., np.diag(covariance)[4:]))/dimensions
            if np.max(relative_sigma) > .8 or yaw_sigma > .9:
                return None
        elif len(views) == 1 and not orientation_held and yaw_sigma > .65:
            return None
        return dict(center=center, yaw=yaw, dimensions=dimensions, covariance=covariance,
                    rms=float(np.sqrt(np.mean(errors**2))), maximum=float(np.max(np.abs(errors))),
                    normalized_rms=float(np.sqrt(np.mean(normalized**2))), yaw_sigma=yaw_sigma,
                    fixed=fixed, motion_constrained=motion_constrained, orientation_held=orientation_held,
                    active_camera_ids=active_camera_ids, metric_camera_ids=metric_camera_ids)

    def _learn(self, fit):
        dimensions = fit['dimensions']
        variance = np.maximum(np.diag(fit['covariance'])[4:], (.025*dimensions)**2)
        if self._dimensions is not None:
            squared = (dimensions-self._dimensions)**2/(variance+self._dimension_variance)
            if np.any(squared > np.array([16., 25., 16.])):
                return False
        self._samples.append((dimensions.copy(), variance.copy()))
        self._sample_count += 1
        values = np.asarray([sample[0] for sample in self._samples])
        variances = np.asarray([sample[1] for sample in self._samples])
        dimensions = np.empty(3)
        uncertainty = np.empty(3)
        for axis in range(3):
            order = np.argsort(values[:, axis])
            weights = 1./variances[order, axis]
            index = np.searchsorted(np.cumsum(weights), weights.sum()/2)
            dimensions[axis] = values[order[index], axis]
            # Correlated silhouette/model error cannot shrink to zero from a
            # burst of nearly identical frames. Width may remain much less
            # certain than the metric height that supports single-view range.
            residual = values[:, axis]-dimensions[axis]
            scatter = float(np.average(residual**2, weights=1./variances[:, axis]))
            uncertainty[axis] = max(1./weights.sum(), scatter,
                                    ([.08, .25, .08][axis]*dimensions[axis])**2)
        self._dimensions = dimensions
        self._dimension_variance = uncertainty
        return True

    def update(self, frame_index, timestamp, observations, prior_cm=None, velocity_cm_s=None):
        if (not isinstance(frame_index, (int, np.integer)) or frame_index < 0
                or (self._frame is not None and frame_index <= self._frame)):
            raise ValueError('Shape frame must be a strictly increasing nonnegative integer')
        timestamp = float(timestamp)
        if (not math.isfinite(timestamp) or timestamp < 0
                or (self._timestamp is not None and timestamp < self._timestamp)):
            raise ValueError('Shape timestamp must be finite and cannot go backwards')
        prior = _vector(prior_cm, 'Position prior')
        prior = None if prior is None else prior/100.
        velocity = _vector(velocity_cm_s, 'Velocity')
        elapsed = 0. if self._timestamp is None else timestamp-self._timestamp
        self._frame, self._timestamp = int(frame_index), timestamp
        all_views = self._views(frame_index, observations)
        if not all_views:
            return None
        full_views = [view for view in all_views if view['learnable']]
        views = full_views if len(full_views) >= 2 else all_views
        seed = self._seed(views)
        # The all-view ray seed may include a partial box that the robust fitter
        # will later reject. It cannot establish body motion or the front sign.
        external_heading = self._motion_heading(timestamp, None, velocity)
        heading = external_heading if external_heading is not None else self._heading

        def robust_fit(candidates, fixed):
            start = self._seed(candidates)
            if start is None:
                return None, candidates
            fit = self._fit(candidates, start, heading, shared=self._dimensions, fixed=fixed)
            if fit is not None or len(candidates) < 3:
                return fit, candidates
            alternatives = []
            for pair in combinations(candidates, 2):
                start = self._seed(pair)
                if start is None:
                    continue
                candidate = self._fit(pair, start, heading, shared=self._dimensions, fixed=fixed)
                if candidate is not None:
                    score = candidate['normalized_rms'] + .1*(2-sum(v['confidence'] for v in pair))
                    alternatives.append((score, candidate, list(pair)))
            if not alternatives:
                return None, candidates
            _, candidate, pair = min(alternatives, key=lambda item: item[0])
            return candidate, pair

        def range_consistent(candidate, view):
            if prior is None:
                return True
            previous_depth = float((prior-view['origin']) @ view['direction'])
            current_depth = float((candidate['center']-view['origin']) @ view['direction'])
            allowance = max(5., .35*max(previous_depth, 1.), 50.*elapsed)
            return previous_depth > 0 and abs(current_depth-previous_depth) <= allowance

        def single_view_fallback(candidates):
            # Only a causal prediction may arbitrate competing single-view
            # ranges. A failed current center triangulation is not evidence for
            # choosing between an intact truck and a tiny foreground fragment.
            if self._dimensions is None or prior is None:
                return None, candidates
            alternatives = []
            for view in candidates:
                if not view['learnable']:
                    continue
                distance = view['fy']*self._dimensions[2]/(view['box'][3]-view['box'][1])
                start = view['origin']+distance*view['direction']
                candidate = self._fit([view], start, heading, shared=self._dimensions, fixed=True)
                if candidate is None or not range_consistent(candidate, view):
                    continue
                scale = max(3., .1*np.linalg.norm(prior-view['origin']), 15.*elapsed)
                score = candidate['normalized_rms']**2 + np.sum((candidate['center']-prior)**2)/scale**2
                alternatives.append((score, candidate, [view]))
            if not alternatives:
                return None, candidates
            _, candidate, selected = min(alternatives, key=lambda item: item[0])
            return candidate, selected

        if len(full_views) >= 2:
            fit, views = robust_fit(full_views, fixed=False)
            if fit is None and self._dimensions is not None:
                fit, views = robust_fit(full_views, fixed=True)
            if fit is None:
                fit, views = single_view_fallback(full_views)
            if fit is None:
                return None
            previous_learning = (self._samples.copy(), self._sample_count,
                                 None if self._dimensions is None else self._dimensions.copy(),
                                 None if self._dimension_variance is None else self._dimension_variance.copy())
            if not fit['fixed']:
                self._learn(fit)
                fit = self._fit(views, fit['center'], heading, shared=self._dimensions, fixed=True)
            if fit is None:
                self._samples, self._sample_count, self._dimensions, self._dimension_variance = previous_learning
                fit, views = single_view_fallback(full_views)
        elif self._dimensions is not None:
            # Initial range follows learned physical height, not the motion prior.
            if seed is None:
                view = views[0]
                height_px = view['box'][3]-view['box'][1]
                distance = view['fy']*self._dimensions[2]/height_px
                seed = view['origin'] + distance*view['direction']
            fit = self._fit(views, seed, heading, shared=self._dimensions, fixed=True)
            if fit is not None and len(views) == 1 and not range_consistent(fit, views[0]):
                fit = None
        else:
            return None
        if fit is None:
            return None
        views = [view for view in views if view['camera_id'] in fit['active_camera_ids']]
        metric_views = [view for view in views if view['camera_id'] in fit['metric_camera_ids']]
        multi_view = self._seed(metric_views) is not None
        partial = any(view['partial'] for view in views)
        center, yaw, dimensions = fit['center'], fit['yaw'], fit['dimensions']
        if external_heading is not None and not fit['orientation_held']:
            self._heading = external_heading
        elif multi_view and not partial:
            observed_heading = self._motion_heading(timestamp, center, None)
            if observed_heading is not None:
                self._heading = observed_heading
        sign_hint = self._heading
        front_observable = sign_hint is not None
        if front_observable and math.cos(yaw-sign_hint) < 0:
            yaw += math.pi
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self._yaw = yaw
        if not partial:
            self._last_full_yaw = yaw
            self._last_full_timestamp = timestamp
        if multi_view and not partial:
            self._centers.append((timestamp, center.copy()))
        covariance = fit['covariance']
        model_floor = (2. if partial else .75)**2
        center_model_floor = np.eye(3)*model_floor
        center_covariance = covariance[:3, :3] + center_model_floor
        if not multi_view:
            # Common scale error is correlated across repeated single-view frames.
            # It must survive the local pixel fit and be exposed to the world filter.
            relative = center-views[0]['origin']
            scale_sigma = max(.08, math.sqrt(self._dimension_variance[2])/self._dimensions[2])
            if self._sample_count < 3:
                scale_sigma = max(scale_sigma, .15)
            if partial:
                scale_sigma = max(scale_sigma, .2)
            scale_systematic = scale_sigma**2*np.outer(relative, relative)
            center_covariance += scale_systematic
        else:
            scale_systematic = np.zeros((3, 3))
        systematic = center_model_floor + scale_systematic
        front = center + dimensions[0]/2 * np.array([math.cos(yaw), math.sin(yaw), 0.])
        front_jacobian = np.zeros((3, covariance.shape[0]))
        front_jacobian[:, :3] = np.eye(3)
        front_jacobian[:, 3] = dimensions[0]/2*np.array([-math.sin(yaw), math.cos(yaw), 0.])
        if not fit['fixed']:
            front_jacobian[:, 4] = .5*np.array([math.cos(yaw), math.sin(yaw), 0.])
        front_covariance = front_jacobian @ covariance @ front_jacobian.T + center_model_floor + scale_systematic
        if fit['fixed']:
            front_covariance += (.06*dimensions[0])**2 * np.outer(
                [math.cos(yaw), math.sin(yaw), 0.], [math.cos(yaw), math.sin(yaw), 0.])
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.])
        # Shared length and a held axis are not fresh independent measurements
        # on each frame. Keep their induced front-offset error after averaging.
        front_systematic = systematic + self._dimension_variance[0]/4*np.outer(forward, forward)
        if fit['orientation_held']:
            front_systematic += (dimensions[0]/2*fit['yaw_sigma'])**2*np.outer(side, side)
        return {
            'position_cm': (center*100).tolist(), 'center_cm': (center*100).tolist(),
            'front_cm': (front*100).tolist() if front_observable else None,
            'covariance_cm2': (center_covariance*10000).tolist(),
            'front_covariance_cm2': (front_covariance*10000).tolist() if front_observable else None,
            'systematic_covariance_cm2': (systematic*10000).tolist(),
            'front_systematic_covariance_cm2': (front_systematic*10000).tolist() if front_observable else None,
            'method': 'cuboid_multiview' if multi_view else 'cuboid_singleview',
            'cameras': [view['camera_id'] for view in views],
            'metric_cameras': [view['camera_id'] for view in metric_views] if multi_view else [],
            'metric_depth_observed': multi_view, 'metric_scale_source': 'current_multiview' if multi_view else 'past_multiview_dimensions',
            'quality': 'shape_multiview' if multi_view else 'shape_constrained_single_view',
            'reference_point': 'cuboid_volume_center', 'front_reference_point': 'cuboid_forward_face_center',
            'front_observable': front_observable, 'range_from_shape': not multi_view,
            'yaw_rad': yaw, 'yaw_sigma_rad': fit['yaw_sigma'],
            'dimensions_cm': (dimensions*100).tolist(), 'dimensions_learned': self._sample_count >= 3,
            'dimension_sigma_cm': (np.sqrt(self._dimension_variance)*100).tolist(),
            'provisional_scale': self._sample_count < 3, 'scale_usable': True,
            'partial_silhouette': partial,
            'shape_samples': self._sample_count, 'fit_rms_px': fit['rms'], 'fit_max_px': fit['maximum'],
            'fit_normalized_rms': fit['normalized_rms'],
            'orientation_constrained_by_motion': fit['motion_constrained'],
            'orientation_held': fit['orientation_held'],
            'uncertainty_m': math.sqrt(float(np.linalg.eigvalsh(center_covariance).max())),
            'assumption': 'Upright rigid vehicle cuboid; forward end follows causal motion; learned scale is reused in single views',
        }
