Download 3DIR from the given url and organize the data according to the following root:

```
../Data 

├── Images
  ├── Backpack
    ├── carry
    ├── ...
├── mask
  ├── Backpack
    ├── carry
├── Objects
  ├── Backpack
  ├── Bag
  ├── ...
├── obj_center
  ├── Backpack
  ├── Bag
  ├── ...
├── smplh_contact_pkl
  ├── Backpack
    ├── carry
    ├── ...
├── Curvature
  ├── Human
  ├── Object
├── smplh_param
├── txt_scripts
  ├── train.txt
  ├── val.txt
  ├── Point_train.txt
  ├── Point_val.txt
```

1. `Images/` and `mask/` contain corresponding interaction images and human-object masks.
2. `Objects/` include the object point cloud and 3D affordance annotation, each one is a `.txt` file with `2048` rows, each row is:
```
-0.08506392 -0.022815626 -0.045057002 0.0 0.0 0.0 0.0 0.8886052 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 
```
The first three are the xyz coordinates of a point, and each subsequent column corresponds to the probability of a certain affordance category.

3. `obj_center/` is the spatial postition of the object center, each one is a `.pkl` file with the shape `1x3`.
4. `smplh_contact_pkl/` involves the human contact annotation, each one is a `.pkl` file with the shape `1x6890`, we provide a simple script to visualize the contact on SMPL, as follows:
```
def visual_contact(pkl_file):
    contact = np.load(pkl_file, allow_pickle=True)
    contact_color = np.array([255.0, 191.0, 0.])
    colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
    for i,data in enumerate(contact):
        if data != 0:
            colors[i] = contact_color
    colors = colors / 255.0
    smpl = o3d.io.read_triangle_mesh('smpl_template.ply')
    smpl.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh('visual_contact.ply', smpl)
```

5. `Curvature/` has pre-calculated geometric curvatures of the posed human body and objects, the curvature of human is `6890 x 1` while object is `2048 x 1`.
6. `smplh_param` is the SMPLH pseudo-parameters, it is stored in `JSON` format, note that we employ the SMPLH which the shape is 16 and the parameters are represented by rotation matrix. As shown in the following example:
```
{
    "image_path": {
        "smplh_param":{
            "shape": [16],
            "body_pose": [21x3x3],
            "left_hand_pose": [15x3x3],
            "right_hand_pose": [15x3x3],
            "global_orient": [3x3], 
            "transl": [3],

        },
        "human_box": [x1,y1,x2,y2]
    }
}
```

7. `txt_scripts/` specifies the data for training and testing.

Due to the licence, we are unable to directly release the data of BEHAVE. If there is a need, please contact `yyuhang@mail.ustc.edu.cn` and we will consider providing the data id we used in LEMON.