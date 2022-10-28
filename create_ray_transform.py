
import numpy as np 
import odl 



def get_ray_trafo(start_angle, stop_angle, impl="astra_cuda"):
    """
    Hyperparameters from Metadata in .mat files

    """    
    
    num_angles= 721
    det_shape = 560

    M = 1.348414746992646 

    DistanceSourceDetector=553.74
    DistanceSourceOrigin=410.66

    DistanceDetectorOrigin = DistanceSourceDetector - DistanceSourceOrigin

    angle_partition = odl.uniform_partition_fromgrid(
                odl.discr.grid.RectGrid(np.linspace(0, 360, 721)*np.pi/180))[start_angle:stop_angle]
                    
    effPixel = 0.1483223173330444

    det_partition = odl.uniform_partition(-M*det_shape/2, M*det_shape/2, det_shape)

    geometry =  odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, det_partition, 
                                                        src_radius=DistanceSourceOrigin/effPixel , 
                                                        det_radius=DistanceDetectorOrigin/effPixel,
                                                        src_to_det_init=(-1, 0))

    space = odl.discr.discr_space.uniform_discr([-256,-256], [256,256], (512, 512), dtype=np.float32)

    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)

    return ray_trafo



if __name__ == "__main__":

    ray_trafo = get_ray_trafo(0, 181)

    print(ray_trafo.range)