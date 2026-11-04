import unittest

from peleng.silhouette import center_observations


class Rig:
    width, height = 1000, 600

    def project(self, camera, frame, point):
        return 500+point[0]*.1, 300-point[2]*.1


class SilhouetteTests(unittest.TestCase):
    def setUp(self):
        self.model = {'dimensions_cm':[1000,400,400], 'yaw_rad':0.}

    def test_intact_view_survives_while_narrow_occluded_fragment_cannot_supply_center_ray(self):
        observations = [
            {'camera_id':1,'bbox':[451,281,549,319]},
            {'camera_id':2,'bbox':[480,282,500,318]},
        ]
        kept, rejected = center_observations(Rig(), 10, observations, self.model, [0,0,0])
        self.assertEqual([view['camera_id'] for view in kept], [1])
        self.assertEqual(rejected[0]['camera_id'], 2)
        self.assertEqual(rejected[0]['reason'], 'partial_extent')
        self.assertEqual(observations[1]['bbox'], [480,282,500,318])

    def test_no_learned_volume_or_no_causal_position_does_not_invent_rejection(self):
        views=[{'camera_id':1,'bbox':[480,282,500,318]}]
        self.assertEqual(center_observations(Rig(),0,views,None,[0,0,0]),(views,[]))
        self.assertEqual(center_observations(Rig(),0,views,self.model,None),(views,[]))

    def test_explicit_partial_flags_do_not_become_center_rays_before_scale_is_known(self):
        views=[{'camera_id':i,'bbox':[450,280,550,320],**flags}
               for i,flags in enumerate([{'quality':{'partial':True}},{'occluded':True},
                                         {'truncated':True}],start=1)]
        kept,rejected=center_observations(Rig(),0,views,None,None)
        self.assertEqual(kept,[])
        self.assertEqual(len(rejected),3)

    def test_uncertain_prediction_cannot_lock_out_reacquisition(self):
        views=[{'camera_id':1,'bbox':[480,282,500,318]}]
        kept,rejected=center_observations(Rig(),10,views,self.model,[0,0,0],position_sigma_cm=3000.)
        self.assertEqual(kept,views)
        self.assertEqual(rejected,[])

    def test_truncated_box_does_not_define_whole_object_center(self):
        views=[{'camera_id':1,'bbox':[0,280,60,320]}]
        _,rejected=center_observations(Rig(),10,views,self.model,[0,0,0])
        self.assertEqual(rejected[0]['reason'],'truncated_center')


if __name__=='__main__':unittest.main()
