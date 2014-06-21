import math

import pygame
from pygame.locals import *
from pygame.sprite import Sprite
import numpy as np
import numpy


image_dir = "/"

scale_dist = 300. / (149600000 * 1000)

sun_mass = 1.9891e30  # kg

speed_mars = 24.13 * 1000


class GlobalSolver():
    def __init__(self):
        self.G = 6.67e-11


    def globel_solve(self, objects, h):
        self.h = h
        self.objects = objects
        self.obj_len = len(objects)
        for i in range(self.obj_len):
            object = objects[i]
            object.pos, object.speed = self.solve(object, i)

    def acc_sum(self, x, nr_obj):
        acc = np.zeros(2)
        for i in range(self.obj_len):
            if i != nr_obj:
                acc += self.acceleration(x, self.objects[i].pos, self.objects[i].mass)

        return acc

    def acceleration(self, my_x, to_x, to_m):
        vector_to_sun = to_x - my_x  # earth located at origin
        # if numpy.linalg.norm(vector_to_sun) == 0:
        # return 0
        return self.G * to_m / numpy.linalg.norm(vector_to_sun) ** 3 * vector_to_sun

        # @abstractmethod
        # def solve(self, obj, i):
        # pass


class RK4(GlobalSolver):
    def solve(self, obj, i):
        k_x = [0] * 5
        k_v = [0] * 5
        k_x[0] = obj.pos
        k_v[0] = obj.speed
        h = self.h

        acc = [0] * 4
        acc[0] = self.acc_sum(k_x[0], i)
        k_x[1] = h * k_v[0]
        k_v[1] = h * acc[0]

        new_x = k_x[0] + k_x[1]
        acc[1] = self.acc_sum(new_x, i)
        k_x[2] = h * (2 * k_v[0] + k_v[1]) / 2
        k_v[2] = h * (2 * acc[0] + acc[1]) / 2

        new_x = k_x[0] + k_x[2]
        acc[2] = self.acc_sum(new_x, i)
        k_x[3] = h * (2 * k_v[0] + k_v[2]) / 2
        k_v[3] = h * (2 * acc[0] + acc[2]) / 2

        new_x = k_x[0] + k_x[3]
        acc[3] = self.acc_sum(new_x, i)
        k_x[4] = h * (2 * k_v[0] + k_v[3])
        k_v[4] = h * (2 * acc[0] + acc[3])

        x = k_x[0] + (k_x[1] + 2 * k_x[2] + 2 * k_x[3] + k_x[4]) / 6
        v = k_v[0] + (k_v[1] + 2 * k_v[2] + 2 * k_v[3] + k_v[4]) / 6

        if (np.isnan(np.sum(x))):
            print "shit"
            return k_x[0], k_v[0]

        return x, v


class RK2(GlobalSolver):
    def solve(self, obj, i):
        x0 = obj.pos
        v0 = obj.speed
        acc1 = self.acc_sum(x0, i)
        x = x0 + self.h * v0
        v = v0 + self.h * acc1
        acc2 = self.acc_sum(x, i)
        x2 = x0 + self.h / 2 * (v0 + v)
        v2 = v0 + self.h / 2 * (acc1 + acc2)
        if (np.isnan(np.sum(x2))):
            return x0, v0
        return x2, v2


class RK1(GlobalSolver):
    def solve(self, obj, i):
        x0 = obj.pos
        v0 = obj.speed
        acc1 = self.acc_sum(obj.pos, i)
        x = x0 + self.h * v0
        v = v0 + self.h * acc1
        if (np.isnan(np.sum(x))):
            return x0, v0
        return x, v


class Object(Sprite):
    def __init__(self, screen, pos, r, mass, name, speed, distance=0):
        self.r = r
        self.name = name
        self.image_scale = 25. / (6.371 * 1000)
        self.dispr = int(r * self.image_scale)
        self.image_size = np.array([self.dispr, self.dispr])
        self.pos = np.array(pos) / scale_dist
        self.mass = mass
        self.image = self.convert_image(name)
        self.base_image = self.image
        self.speed = np.array(speed)
        self.rect = self.image.get_rect()
        self.screen = screen

    def convert_image(self, name):
        image = pygame.image.load(name).convert_alpha()
        return pygame.transform.scale(image, (self.dispr, self.dispr))

    def bitme(self):
        rescaled = self.pos * scale_dist - self.image_size / 2
        move = self.image.get_rect().move(rescaled)
        self.screen.blit(self.image, move)


class Sun(Object):
    def __init__(self, screen, pos):
        super(Sun, self).__init__(screen, pos, 6.371 * 1000 * 5, 1.9891e30, "sun.png", [0., 0.])


class Earth(Object):
    def __init__(self, screen, pos):
        radius = 149600000 * 1000
        total_time = 365 * 24 * 3600
        speed = 2.0 * math.pi * radius / total_time
        distance = radius * scale_dist
        pos[0] += distance
        super(Earth, self).__init__(screen, pos, 6.371 * 1000, 5.97219 * 10 ** 24, "earth.png",
                                    [0., speed])


class Mars(Object):
    def __init__(self, screen, pos):
        distance = 227900000 * 1000 * scale_dist
        pos[0] += distance
        super(Mars, self).__init__(screen, pos, 3.390 * 1000, 639E21, "mars.png",
                                   [0., speed_mars])


class Ship(Object):
    def __init__(self, screen, pos):
        radius = 149600000 * 1000
        total_time = 365 * 24 * 3600
        speed = 2.0 * math.pi * radius / total_time
        distance = radius * scale_dist
        pos[0] += distance * 0.99
        super(Ship, self).__init__(screen, pos, 3.390 * 1000, 2000e3, "ship.png",
                                   [0., speed * 0.95])


class Moon(Object):
    def __init__(self, screen, pos):
        earth_radius = 149600000 * 1000
        total_time = 365 * 24 * 3600
        earth_speed = 2.0 * math.pi * earth_radius / total_time
        distance = (384400 * 1000 + earth_radius) * scale_dist
        pos[0] += distance
        super(Moon, self).__init__(screen, pos, 1.737 * 1000, 7.37219e22, "moon.png",
                                   [0., earth_speed + 1 * 1000])


class App:
    def __init__(self):
        self._running = True
        self.screen = None
        self._image_surf = None

    def on_init(self):
        pygame.init()
        self.size = [1400, 600]
        self.half_size = [self.size[0] / 2., self.size[1] / 2.]
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE)
        self._running = True
        self.frame_rate = 70
        self.steps = 3
        # self.clock.tick(self.frame_rate)
        self.objects = [Sun(self.screen, list(self.half_size))]
        self.objects.append(Earth(self.screen, list(self.half_size)))
        self.objects.append(Mars(self.screen, list(self.half_size)))
        self.objects.append(Ship(self.screen, list(self.half_size)))
        self.objects.append(Moon(self.screen, list(self.half_size)))
        # self.objects.append(Ship(self.screen, list(self.half_size)))
        self.solver = RK4()


    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        multi = 400
        time_passed = multi * self.clock.tick(self.frame_rate)
        for i in range(self.steps):
            for obj in self.objects:
                self.solver.globel_solve(self.objects, time_passed * 1. / self.steps)
        self.screen.fill((0, 0, 0))
        for obj in self.objects:
            obj.bitme()
        pygame.display.flip()
        self.clock.tick(self.frame_rate)


    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        while ( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()