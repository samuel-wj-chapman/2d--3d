import bpy
import os
import math
import random

import os

# Use os.path.expanduser to expand the tilde to the full home directory path
stl_directory = os.path.expanduser("~/Documents/dataset/Mushroom/files/")
stl_file_paths = [os.path.join(stl_directory, f) for f in os.listdir(stl_directory) if f.endswith('.stl')]
output_directory = os.path.expanduser("~/Documents")
num_images_per_object = 10
view_coverage_range = (0.7, 0.9)  # Object coverage in the camera view

# Function to delete all objects in the scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Function to ensure a camera exists and is set as the active camera for the scene
def setup_camera():
    # Check if there is a camera in the scene
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    
    # If no camera was found, create one
    if camera is None:
        bpy.ops.object.camera_add(location=(0, -3, 1))
        camera = bpy.context.object
    
    # Set the newly added or found camera as the active camera
    bpy.context.scene.camera = camera

    return camera

def setup_lights():
    light_data = bpy.data.lights.new(name="Soft_Light", type='AREA')
    light_data.energy = 1000  # Adjust energy as needed for your scene

    for angle in range(0, 360, 45):  # Creates lights in a circular pattern
        light_object = bpy.data.objects.new(name="Soft_Light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (10 * math.sin(math.radians(angle)), 10 * math.cos(math.radians(angle)), 10)
        light_object.rotation_euler = (math.radians(45), 0, math.radians(angle))


# Function to load an STL file
def load_stl(file_path):
    bpy.ops.import_mesh.stl(filepath=file_path)

# Function to position the camera based on object bounds
def position_camera(obj, coverage):
    # Calculate the camera distance from the object
    dimensions = obj.dimensions
    max_dimension = max(dimensions)
    camera_distance = (0.5 * max_dimension) / math.tan(math.radians(bpy.context.scene.camera.data.angle_x / 2))

    # Adjust for coverage
    coverage_factor = random.uniform(*coverage)
    camera_distance /= coverage_factor

    # Set camera location to maintain the object in view
    bpy.context.scene.camera.location = (0, -camera_distance, camera_distance)
    bpy.context.scene.camera.rotation_euler = (math.radians(45), 0, 0)


# Function to render and save an image
def render_save_image(obj_name, angle_x, angle_y, output_path, image_id):
    # Set object rotation
    bpy.context.scene.objects[obj_name].rotation_euler = (math.radians(angle_x), 0, math.radians(angle_y))

    # Update scene
    bpy.context.view_layer.update()

    # Render and save image
    file_name = f"{obj_name}_angleX{angle_x}_angleY{angle_y}_{image_id}.png"
    bpy.context.scene.render.filepath = os.path.join(output_path, file_name)
    bpy.ops.render.render(write_still=True)

# Setup lighting
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 10))

# Main processing loop
for stl_path in stl_file_paths:
    clear_scene()
    setup_lights()  # Set up surround lights
    load_stl(stl_path)
    camera = setup_camera()
    obj = bpy.context.selected_objects[0]
    position_camera(obj, view_coverage_range)


    for i in range(num_images_per_object):
        angle_x = random.randint(0, 360)
        angle_y = random.randint(0, 360)
        render_save_image(obj.name, angle_x, angle_y, output_directory, i)

print("Dataset generation complete.")
