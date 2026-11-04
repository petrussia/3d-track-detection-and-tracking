"""Independent pinhole side-view fixtures for causal leading contact."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from peleng.geometry import CameraRig
try:
    from peleng.initialization import LeadingContact
except ModuleNotFoundError as exc:
    if exc.name != 'peleng.initialization':
        raise
    LeadingContact = None


class LeadingContactTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(LeadingContact, 'Leading-contact initialization is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.make_rig()

    def make_rig(self, rotating=False):
        # Camera at(0,0,1000), forward+X. fx=fy=1000: ground atX10000 hasv640.
        text = ('Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;10\n'
                '0;0;0;1000;0;0;0;;;матрица;19,2*10,8мм\n')
        if rotating:
            text += '3;0;0;1000;0;0;1;;;;\n'
        (Path(self.temp.name)/'object_transform_CineCameraActor7.csv').write_text(text, encoding='cp1251')
        self.rig = CameraRig(self.temp.name)
        self.helper = LeadingContact(self.rig)

    @staticmethod
    def base():
        return {'position_cm':[10000.,0.,0.], 'method':'ground_plane', 'cameras':[7],
                'confidence':.6, 'reference_point':'approximate_bbox_ground_contact',
                'covariance_cm2':np.eye(3).tolist(), 'metric_depth_observed':False}

    def observation(self, frame, direction=1, vertical=0, **extra):
        # World front contact Y=+1000+100*f forright, -1000-100*f forleft.
        center=960+direction*10*frame
        return {'camera_id':7, 'bbox':[center-100,580+vertical*frame,center+100,640+vertical*frame],
                'confidence':.6, **extra}

    def advance(self, direction=1, vertical=0):
        result=None
        for f in range(4):
            result=self.helper.update(f,f/30,[self.observation(f,direction,vertical)],self.base())
        return result

    def test_side_motion_uses_projected_leading_ground_point_not_box_center(self):
        result=self.advance()
        np.testing.assert_allclose(result['position_cm'],[10000.,1300.,0.],atol=1e-8)
        self.assertEqual(result['reference_point'],'front_ground_contact')
        self.assertFalse(result['metric_depth_observed'])
        self.assertEqual(result['method'],'ground_plane')
        self.assertGreater(np.linalg.eigvalsh(result['covariance_cm2']).min(),1.)
        self.assertEqual(result['confidence'],.6)
        self.assertEqual(self.base()['position_cm'],[10000.,0.,0.])

    def test_reversed_motion_selects_other_ground_corner(self):
        np.testing.assert_allclose(self.advance(-1)['position_cm'],[10000.,-1300.,0.],atol=1e-8)

    def test_unknown_static_and_vertical_motion_never_invent_front(self):
        for horizontal,vertical in [(0,0),(0,10),(1,15)]:
            self.helper.reset()
            result=self.advance(horizontal,vertical)
            self.assertEqual(result,self.base())

    def test_partial_clipped_crop_and_end_on_views_do_not_supply_front(self):
        self.advance()
        for extra in [{'quality':{'partial':True}}, {'crop_only':True}, {'bbox':[10,500,50,650]},
                      {'bbox':[0,580,200,640]}]:
            self.helper.reset()
            for f in range(4):
                o=self.observation(f);o.update(extra)
                result=self.helper.update(f,f/30,[o],self.base())
            self.assertEqual(result,self.base())

    def test_camera_pan_invalidates_motion_history(self):
        self.make_rig(rotating=True)
        self.assertEqual(self.advance(),self.base())

    def test_confirmed_direction_survives_brief_pixel_jitter_then_expires(self):
        self.advance()
        result=self.helper.update(4,4/30,[self.observation(3)],self.base())
        self.assertEqual(result['reference_point'],'front_ground_contact')
        for f in range(5,30):
            result=self.helper.update(f,f/30,[self.observation(3)],self.base())
        self.assertEqual(result,self.base())

    def test_order_is_checked_and_reset_removes_old_motion(self):
        self.advance()
        with self.assertRaises(ValueError):
            self.helper.update(2,.02,[self.observation(2)],self.base())
        self.helper.reset()
        self.assertEqual(self.helper.update(0,0,[self.observation(0)],self.base()),self.base())

    def test_only_selected_ground_camera_can_change_reference(self):
        base=self.base();base['method']='triangulated'
        for f in range(4):
            result=self.helper.update(f,f/30,[self.observation(f)],base)
        self.assertEqual(result,base)
        self.helper.reset()
        base=self.base();base['cameras']=[99]
        for f in range(4):
            result=self.helper.update(f,f/30,[self.observation(f)],base)
        self.assertEqual(result,base)

    def test_reversal_requires_new_consistent_motion_and_then_flips(self):
        self.advance()
        for f in range(4,24):
            result=self.helper.update(f,f/30,[self.observation(6-f)],self.base())
        self.assertEqual(result['image_motion_sign'],-1)
        self.assertLess(result['position_cm'][1],0.)

    def test_long_gap_does_not_reuse_old_direction(self):
        self.advance()
        result=self.helper.update(60,2.,[self.observation(3)],self.base())
        self.assertEqual(result,self.base())

    def test_missing_base_still_accumulates_current_image_motion(self):
        for f in range(3):
            self.assertIsNone(self.helper.update(f,f/30,[self.observation(f)],None))
        result=self.helper.update(3,.1,[self.observation(3)],self.base())
        self.assertEqual(result['reference_point'],'front_ground_contact')
