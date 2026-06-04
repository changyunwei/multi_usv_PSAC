import pyglet
from pathlib import Path


def center_image(image):
    image.anchor_x = image.width/2
    image.anchor_y = image.height/2


pyglet.resource.path = [str(Path(__file__).resolve().parent / "resources")]
pyglet.resource.reindex()


kernel_image = pyglet.resource.image("kernel.png")
center_image(kernel_image)

guard_image = pyglet.resource.image("guard.png")
center_image(guard_image)

threat_image = pyglet.resource.image("threat.png")
center_image(threat_image)
