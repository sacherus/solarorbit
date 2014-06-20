import math

import pygame
from pygame.locals import *
from pygame.sprite import Sprite
import numpy as np
import numpy


image_dir = "/"

black = (   0, 0, 0)
white = ( 255, 255, 255)
green = (   0, 255, 0)
red = ( 255, 0, 0)
frame_rate = 40
scale = 50. / (6.371 * 1000)
scale_dist = 300. / (149600000 * 1000)
avg_size = (100, 100)

earth_mass = 1.9891e30  # kg
radius = 149600000 * 1000
total_time = 365 * 24 * 3600
speed = 2.0 * math.pi * radius / total_time

speed_mars = 24.13 * 1000

print speed * scale_dist

sun_position = np.array([600, 400]) / scale_dist


class Solver():
    def __init__(self):
        self.G = 6.67e-11

    def acceleration(self, object_position):
        vector_to_sun = sun_position - object_position  # earth located at origin

        return self.G * earth_mass / numpy.linalg.norm(vector_to_sun) ** 3 * vector_to_sun


    def solve_with_pixels(self, x0, v0, h):
        x, v = self.solve(np.array(x0) / scale_dist, np.array(v0) / scale_dist, h)
        return (x * scale_dist).tolist(), (v * scale_dist).tolist()

    def solve(self, x0, v0, h):
        self.h = h
        xe = x0 + self.h * v0
        ve = v0 + self.h * self.acceleration(x0)
        x = x0 + self.h * (v0 + ve) / 2
        v = v0 + self.h * (self.acceleration(x0) + self.acceleration(xe)) / 2
        if (np.isnan(np.sum(x))):
            return x0, v0
        return x, v


class Object(Sprite):
    def __init__(self, screen, pos, r, mass, name, speed, distance=0):
        self.r = r
        self.dispr = int(r * scale)
        self.image_size = [self.dispr, self.dispr]
        self.pos = pos
        self.mass = mass
        self.image = self.convert_image(name)
        self.base_image = self.image
        self.speed = speed
        self.rect = self.image.get_rect()
        self.screen = screen
        self.solver = Solver()

    def convert_image(self, name):
        image = pygame.image.load(name).convert_alpha()
        return pygame.transform.scale(image, (self.dispr, self.dispr))

    def bitme(self):
        move = self.image.get_rect().move(self.pos[0] - self.image_size[0] / 2, self.pos[1] - self.image_size[1] / 2)
        self.screen.blit(self.image, move)


    def update(self, time_passed):
        # self.pos, self.speed = self.solver(time_passed)
        self.pos, self.speed = self.solver.solve_with_pixels(self.pos, self.speed, time_passed)
        # self.pos = [self.pos[0] + self.speed[0] * time_passed, self.pos[1] + self.speed[1] * time_passed]


# 100 pixels are one sun



class Sun(Object):
    def __init__(self, screen, pos):
        super(Sun, self).__init__(screen, pos, 6.9550 * 1000, 1.9891 * 10 ** 30, "sun.png", [0., 0.])


class Earth(Object):
    def __init__(self, screen, pos):
        distance = radius * scale_dist
        pos[0] += distance
        super(Earth, self).__init__(screen, pos, 6.371 * 1000, 5.97219 * 10 ** 24, "earth.png",
                                    [0., speed * scale_dist])


class Mars(Object):
    def __init__(self, screen, pos):
        distance = 227900000 * 1000 * scale_dist
        pos[0] += distance
        super(Mars, self).__init__(screen, pos, 3.390 * 1000, 5.97219 * 10 ** 24, "mars.png",
                                   [0., speed_mars * scale_dist])



class Ship(Object):
    def __init__(self, screen, pos):
        distance = 227900000 * 1000 * scale_dist
        pos[0] += distance
        super(Ship, self).__init__(screen, pos, 3.390 * 1000, 5.97219 * 10 ** 24, "ship.png",
                                   [0., speed_mars * scale_dist])


# class Moon(Object):
# def __init__(self, x, y):
# super(Moon, self).__init__(x, y, 6.371 * 1000, 5.97219 * 10 ** 24, "moon.png", [0.1, 0.1])


class App:
    def __init__(self):
        self._running = True
        self.screen = None
        self._image_surf = None

    def on_init(self):
        pygame.init()
        self.size = [1200, 800]
        self.half_size = [self.size[0] / 2., self.size[1] / 2.]
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.size, pygame.HWSURFACE)
        self._running = True
        self.clock.tick(frame_rate)
        self.objects = [Sun(self.screen, list(self.half_size))]
        self.objects.append(Earth(self.screen, list(self.half_size)))
        self.objects.append(Mars(self.screen, list(self.half_size)))
        # )), Mars(self.screen, list(self.half_size))]


    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        # multi = 10
        time_passed = 10000 * self.clock.tick(50)
        self.screen.fill(black)
        for obj in self.objects:
            obj.update(time_passed)
            obj.bitme()
        pygame.display.flip()
        self.clock.tick(frame_rate)


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