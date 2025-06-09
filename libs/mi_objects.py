import mitsuba as mi


class MiScene(object):
    def __init__(self):
        self.scene = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": -1},
        }
        return

    def dict(self):
        return self.scene
    
    def add(self, key, object):
        self.scene[key] = object.dict()
        return


class MiSensor(object):
    def __init__(
        self,
        origin: list,
        target: list,
        fov: int = 25,
        sample_count: int = 256,
        film_width: int = 540,
        film_height: int = 540,
    ):
        self.origin = origin
        self.target = target
        self.fov = fov
        self.sample_count = sample_count
        self.film_width = film_width
        self.film_height = film_height
        return

    def dict(self):
        sensor = {
            "type": "perspective",
            "fov": self.fov,
            "near_clip": 0.1,
            "far_clip": 100.0,
            "to_world": mi.Transform4f().look_at(
                origin=self.origin, target=self.target, up=[0, 0, 1]
            ),
            "sampler": {"type": "ldsampler", "sample_count": self.sample_count},
            "film": {
                "type": "hdrfilm",
                "width": self.film_width,
                "height": self.film_height,
                "rfilter": {"type": "gaussian"},
            },
        }

        return sensor


class MiFloor(object):
    def __init__(self, width: int = 10, height: int = 10, color: list = [1, 1, 1]):
        self.width = width
        self.height = height
        self.color = color
        return

    def dict(self):
        floor = {
            "type": "rectangle",
            "to_world": mi.Transform4f()
            .scale([self.width, self.height, 1])
            .translate([0, 0, -0.5]),
            "bsdf": {
                "type": "roughplastic",
                "distribution": "ggx",
                "alpha": 0.05,
                "int_ior": 1.46,
                "diffuse_reflectance": {"type": "rgb", "value": self.color},
            },
        }

        return floor


class MiSoftlight(object):
    def __init__(
        self,
        origin: list,
        target: list,
        intensity: int,
        width: int = 10,
        height: int = 10,
    ):
        self.origin = origin
        self.target = target
        self.intensity = intensity
        self.width = width
        self.height = height
        return

    def dict(self):
        light = {
            "type": "rectangle",
            "to_world": mi.Transform4f()
            .look_at(origin=self.origin, target=self.target, up=[0, 0, 1])
            .scale([self.width, self.height, 1]),
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": self.intensity},
            },
        }

        return light
    
class MiSphere(object):
    def __init__(self, center: list, radius: float, color: list):
        self.center = center
        self.radius = radius
        self.color = color
        return

    def dict(self):
        sphere = {
            "type": "sphere",
            "center": self.center,
            "radius": self.radius,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": self.color},
            },
        }

        return sphere
