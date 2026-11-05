"""Map image-derived volume centres to a named physical object reference.

The offset comes from a previously observed object volume. It is never fitted
to target coordinates. A centre bearing becomes a translated bearing line for
the front point; changing only the output XYZ would be a wrong measurement.
"""
import math

import numpy as np


def center_offset(model, reference):
    if reference not in {'front', 'center'}:
        raise ValueError('Reference must be front or center')
    if reference == 'center' or not model.get('front_observable') or model.get('front_cm') is None:
        return np.zeros(3)
    return np.asarray(model['front_cm'], dtype=float)-np.asarray(model['center_cm'], dtype=float)


def orient_reference(model, velocity_cm_s, velocity_covariance, dt):
    """Advance a fallback axis from independent body motion, with SNR gating.

    A fitted current cuboid axis remains preferable. This only supplies a causal
    axis when a complete current silhouette is unavailable; no front-point
    velocity is used to infer the body direction.
    """
    result=dict(model)
    if dt<=0 or not model.get('front_observable') or model.get('front_cm') is None:
        return result
    old_sigma=float(model.get('yaw_sigma_rad',.2))
    result['yaw_sigma_rad']=math.sqrt(old_sigma**2+.35**2*dt)
    if velocity_cm_s is None or velocity_covariance is None:
        return result
    velocity=np.asarray(velocity_cm_s,dtype=float)[:2]
    covariance=np.asarray(velocity_covariance,dtype=float)
    if velocity.shape!=(2,) or covariance.shape!=(2,2) or not np.isfinite(velocity).all() or not np.isfinite(covariance).all():
        return result
    speed=float(np.linalg.norm(velocity))
    if speed < max(100.,3*math.sqrt(max(0.,float(np.linalg.eigvalsh(covariance).max())))):
        return result
    gradient=np.array([-velocity[1],velocity[0]])/speed**2
    yaw_sigma=math.sqrt(max(0.,float(gradient@covariance@gradient))+.2**2)
    if yaw_sigma>.5:
        return result
    offset=center_offset(model,'front')
    old_yaw=math.atan2(offset[1],offset[0])
    desired=math.atan2(velocity[1],velocity[0])
    delta=math.atan2(math.sin(desired-old_yaw),math.cos(desired-old_yaw))
    yaw=old_yaw+float(np.clip(delta,-1.5*dt,1.5*dt))
    remaining=math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
    yaw_sigma=math.sqrt(yaw_sigma**2+remaining**2)
    length=float(np.linalg.norm(offset[:2]))
    new_offset=np.array([length*math.cos(yaw),length*math.sin(yaw),offset[2]])
    result.update(front_cm=(np.asarray(model['center_cm'])+new_offset).tolist(),yaw_rad=yaw,
                  yaw_sigma_rad=yaw_sigma,orientation_source='independent_body_motion')
    return result


def reference_from_body_state(body, covariance_cm2, model, reference, model_age_s=0.):
    """Derive the output point from ONE accepted state; never filter it again.

The body filter owns timeouts, acceptance and reacquisition. Model uncertainty
is retained when translating its position to the cab. No model means no front.
"""
    if reference not in {'front', 'center'}:
        raise ValueError('Reference must be front or center')
    result = dict(body)
    result.update(reference_point=None, object_center_cm=None, front_cm=None,
                  body_measurement_accepted=body['measurement_accepted'],
                  body_rejection_reason=body.get('rejection_reason'),
                  body_raw_measurement=body.get('raw_measurement'),
                  reference_state_source='derived_from_body')
    metric_body = body.get('reference_point') in {'body_center', 'cuboid_center'}
    if body.get('position_cm') is None or not metric_body:
        result.update(state='lost', position_cm=None, velocity_cm_s=None,
                      measurement_accepted=False, method=None, raw_measurement=None)
        return result
    result['object_center_cm'] = list(body['position_cm'])
    if model:
        model = dict(model, yaw_sigma_rad=math.hypot(float(model.get('yaw_sigma_rad', .2)),
                                                    .35*max(0., model_age_s)))
    measurement = dict(position_cm=body['position_cm'], method=body['method'],
                       covariance_cm2=np.asarray(covariance_cm2).tolist())
    # A bearing has already updated the body here; this is a translation of
    # posterior coordinates, not a second bearing observation.
    measurement['method'] = 'body_posterior'
    mapped = reference_from_center(measurement, model, reference) if model else None
    if reference == 'center' and mapped is None:
        mapped = dict(position_cm=body['position_cm'], center_cm=body['position_cm'],
                      front_cm=None, reference_point='cuboid_center',
                      covariance_cm2=np.asarray(covariance_cm2).tolist())
    if mapped is None:
        result.update(state='lost', position_cm=None, velocity_cm_s=None,
                      method=None, measurement_accepted=False, raw_measurement=None,
                      reference_unavailable_reason='front_orientation_unobservable')
        return result
    result.update(position_cm=mapped['position_cm'], front_cm=mapped['front_cm'],
                  reference_point=mapped['reference_point'],
                  uncertainty_m=math.sqrt(max(0., float(np.linalg.eigvalsh(
                      mapped['covariance_cm2']).max())))/100.)
    if result.get('predicted_position_cm') is not None:
        offset = np.asarray(mapped['position_cm']) - np.asarray(body['position_cm'])
        result['predicted_position_cm'] = (np.asarray(result['predicted_position_cm'])+offset).tolist()
    # Translational body velocity alone is not the velocity of a rotating cab.
    if reference == 'front':
        result['velocity_cm_s'] = None
        result['velocity_unavailable_reason'] = 'cab_angular_velocity_not_observed'
    raw = body.get('raw_measurement')
    result['raw_measurement'] = (reference_from_center(raw, model, reference)
                                 if raw and model else raw if reference == 'center' else None)
    return result


def select_reference(shape, reference):
    if reference not in {'front', 'center'}:
        raise ValueError('Reference must be front or center')
    if reference == 'front' and (not shape.get('front_observable') or shape.get('front_cm') is None):
        return None
    result = dict(shape)
    use_front = reference == 'front' and shape.get('front_observable') and shape.get('front_cm') is not None
    result['position_cm'] = list(shape['front_cm'] if use_front else shape['center_cm'])
    if use_front and shape.get('front_covariance_cm2') is not None:
        result['covariance_cm2'] = shape['front_covariance_cm2']
    if use_front and shape.get('front_systematic_covariance_cm2') is not None:
        result['systematic_covariance_cm2']=shape['front_systematic_covariance_cm2']
    result['reference_point'] = 'front_face_center' if use_front else 'cuboid_center'
    return result


def reference_from_center(measurement, model, reference):
    if measurement is None:
        return None
    if reference not in {'front', 'center'}:
        raise ValueError('Reference must be front or center')
    if reference == 'front' and (not model.get('front_observable') or model.get('front_cm') is None):
        return None
    result = dict(measurement)
    offset = center_offset(model, reference)
    center = np.asarray(measurement['position_cm'], dtype=float).copy()
    dimensions = np.asarray(model.get('dimensions_cm', [0.,0.,0.]), dtype=float)
    covariance = np.asarray(measurement.get('covariance_cm2',np.eye(3)*150.**2), dtype=float).copy()
    systematic = np.asarray(measurement.get('systematic_covariance_cm2',np.zeros((3,3))),dtype=float).copy()
    if measurement['method'] == 'ground_plane':
        # A named volume centre lies above its contact proxy by half the learned
        # height. Terrain remains an explicit assumption with its existing bias.
        center[2] += dimensions[2]/2
        covariance[2,2] += (max(50.,.1*dimensions[2]))**2
        systematic[2,2] += (max(50.,.1*dimensions[2]))**2
        result['assumption'] = 'Learned box height above assumed z=0 ground; terrain is not observed'
    front_observable = bool(model.get('front_observable') and model.get('front_cm') is not None)
    front = center + (np.asarray(model['front_cm'])-np.asarray(model['center_cm'])) if front_observable else None
    use_front = reference == 'front' and front_observable
    if use_front:
        length = float(np.linalg.norm(offset))
        forward = offset/max(length,1e-9)
        side = np.array([-forward[1],forward[0],0.])
        yaw_sigma = max(.1, float(model.get('yaw_sigma_rad', .2)))
        offset_covariance = max(75., .06*dimensions[0])**2*np.outer(forward,forward)
        offset_covariance += (length*yaw_sigma)**2*np.outer(side,side)
        covariance += offset_covariance
        # This reference offset reuses the same learned size and held axis.
        # Its error cannot disappear when the same image geometry is repeated.
        systematic += offset_covariance
        if measurement['method'] == 'bearing_update':
            result['origin_cm'] = (np.asarray(measurement['origin_cm'])+offset).tolist()
            result['transverse_sigma_cm'] = math.sqrt(
                measurement['transverse_sigma_cm']**2+float(np.linalg.eigvalsh(offset_covariance).max()))
    result.update(position_cm=(front if use_front else center).tolist(),
                  center_cm=center.tolist(), front_cm=front.tolist() if front is not None else None,
                  front_observable=front_observable, dimensions_cm=dimensions.tolist(),
                  yaw_rad=model.get('yaw_rad'), yaw_sigma_rad=model.get('yaw_sigma_rad'),
                  covariance_cm2=covariance.tolist(),
                  systematic_covariance_cm2=systematic.tolist(),
                  reference_point='front_face_center' if use_front else 'cuboid_center',
                  reference_model_applied=True, metric_scale_source='past_multiview_dimensions')
    return result
