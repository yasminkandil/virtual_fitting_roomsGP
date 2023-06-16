import pywavefront
import os

def write_with_texture(obj, mtl, texture):
    # Set the paths for the input .obj file, the output .mtl file, and the texture file
    obj_path = obj
    mtl_path = mtl
    texture_path = texture

    # Load the .obj file using PyWavefront
    scene = pywavefront.Wavefront(obj_path, create_materials=True)

    # Extract the material information from the scene and write it to the output .mtl file
    with open(mtl_path, "w") as f:
        for material in scene.materials.values():
            f.write("newmtl {}\n".format(material.name))
            f.write("Ka {} {} {}\n".format(*material.ambient))
            f.write("Kd {} {} {}\n".format(*material.diffuse))
            f.write("Ks {} {} {}\n".format(*material.specular))
            f.write("Ns {}\n".format(material.shininess))
            f.write("map_Kd {}\n".format(texture_path))

    # Modify the vertex definitions in the .obj file to include proper texture coordinates
    with open(obj_path, "r") as f:
        obj_data = f.readlines()

    with open(obj_path, "w") as f:
        vertex_count = 1
        texture_coords = []
        vertex_normals = []
        for line in obj_data:
            if line.startswith("v "):
                # Vertex definition: v x y z
                # Write the vertex definition as it is
                f.write(line)
            elif line.startswith("vn "):
                # Vertex normal definition: vn x y z
                # Store the vertex normals to handle missing vertex normals
                normal_coords = line.split()[1:]
                normal_coords = [float(coord) for coord in normal_coords]
                vertex_normals.append(normal_coords)
                f.write(line)
            elif line.startswith("vt "):
                # Texture coordinate definition: vt u v [w]
                # Store the texture coordinates to handle missing texture coordinates
                tex_coords = line.split()[1:]
                tex_coords = [float(coord) for coord in tex_coords]
                texture_coords.append(tex_coords)
                f.write(line)
            elif line.startswith("f "):
                # Face definition: f v1[/vt1][/vn1] v2[/vt2][/vn2] v3[/vt3][/vn3] ...
                # Modify the face definition to include correct texture coordinates and vertex normals
                face_data = line.split()
                new_face_data = []
                for vertex in face_data[1:]:
                    vertex_data = vertex.split("/")
                    vertex_index = int(vertex_data[0])
                    texture_index = vertex_count  # Use the current vertex count as the texture index
                    normal_index = vertex_count  # Use the current vertex count as the normal index
                    if len(vertex_data) >= 2:
                        texture_index = int(vertex_data[1])
                    if len(vertex_data) == 3:
                        normal_index = int(vertex_data[2])
                    if texture_index < 0:
                        # Handle negative texture indices (e.g., -1 represents the last texture coordinate)
                        texture_index = len(texture_coords) + texture_index + 1
                    if normal_index < 0:
                        # Handle negative normal indices (e.g., -1 represents the last normal coordinate)
                        normal_index = len(vertex_normals) + normal_index + 1
                    if texture_index > len(texture_coords):
                        # Handle missing texture coordinates
                        texture_index = 1
                    if normal_index > len(vertex_normals):
                        # Handle missing vertex normals
                        normal_index = 1
                    new_vertex_data = "{}/{}/{}".format(vertex_index, texture_index, normal_index)
                    new_face_data.append(new_vertex_data)
                    vertex_count += 1
                f.write("f {}\n".format(" ".join(new_face_data)))
            else:
                f.write(line)
def editMTLHuman(gender):  
    mtl_file_path = 'E:/GP/fitmoi_mob_app/assets/body_1.mtl'
    with open(mtl_file_path, 'r') as f:
        mtl_contents = f.readlines()
    if gender=='female':
        for i, line in enumerate(mtl_contents):
            if line.startswith('map_Kd'):
                mtl_contents[i] = f'map_Kd femmm_1.png\n'
            if line.startswith('map_Ka'):
                mtl_contents[i] = f'map_Ka femmm_1.png\n'
            if line.startswith('map_Ks'):
                mtl_contents[i] = f'map_Ks femmm_1.png\n'


        with open(mtl_file_path, 'w') as f:
            f.writelines(mtl_contents)
    elif gender=='male':
        for i, line in enumerate(mtl_contents):
            if line.startswith('map_Kd'):
                mtl_contents[i] = f'map_Kd body_1.jpg\n'
            if line.startswith('map_Ka'):
                mtl_contents[i] = f'map_Ka body_1.jpg\n'
            if line.startswith('map_Ks'):
                mtl_contents[i] = f'map_Ks body_1.jpg\n'

        with open(mtl_file_path, 'w') as f:
            f.writelines(mtl_contents)