# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object
    # 作为多次生成刚性物体对象的示例，我们创建其父级 Xform 对象， /World/Origin{i} ，它们对应不同的生成位置。
    # 当将正则表达式 /World/Origin*/Cone 传递给 assets.RigidObject 类时，它会在每个 /World/Origin{i} 位置生成刚性物体对象。
    # 例如，如果场景中存在 /World/Origin1 和 /World/Origin2 ，
    # 则刚性物体对象会分别生成在位置 /World/Origin1/Cone 和 /World/Origin2/Cone 上。
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),    # 设置物体的位置、速度、姿态等，是相对于世界坐标系的。可以通过cone_object.data.default_root_state获取。
    )
    # cone_object是一个物理句柄（这里是包含了4个cone对象），用于获取和设置对象的物理状态。
    # 例如，我们可以使用cone_object.data.default_root_state获取对象的默认根状态，
    cone_object = RigidObject(cfg=cone_cfg)
    """
    和上面与 Spawn Objects 教程中刚性锥体类似的生成配置创建一个圆锥形刚性物体。
    唯一的区别是现在我们将生成配置封装到 assets.RigidObjectCfg 类中。
    该类包含有关资产生成策略、默认初始状态和其他元信息的信息。当将此类传递给 assets.RigidObject 类时，
    当播放模拟时，它会生成对象并初始化相应的物理句柄。
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(), # 添加刚体属性，使其能够与环境交互
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),   # 添加质量属性，使其能够与重力交互
        collision_props=sim_utils.CollisionPropertiesCfg(), # 添加碰撞属性，使其能够与其他物体交互
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)), # 添加可视化材质，使其能够在场景中显示颜色
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )
    """

    # return the scene information
    # 我们将所有场景实体存储在一个字典中，以便在主函数中访问它们。
    # 因为我们想要与刚性物体交互，我们将此实体传递回主函数。
    # 然后，该实体用于在模拟循环中与刚性物体交互。
    # 在后续教程中，我们将看到如何使用 scene.InteractiveScene 类更方便地处理多个场景实体。
    scene_entities = {"cone": cone_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cone_object = entities["cone"]
    # Define simulation stepping
    # 我们使用 sim.get_physics_dt() 方法获取模拟的时间步长。
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            '''
            为了重置生成的刚性物体对象的模拟状态，我们需要设置其姿势和速度。
            它们共同定义了生成的刚性物体对象的根状态。重要的是要注意，此状态定义在 模拟世界坐标系 中，而不是它们父级 Xform 对象的坐标系。
            这是因为物理引擎只能理解世界坐标系，而不能理解父级 Xform 对象的坐标系。
            因此，在设置之前，我们需要将刚性物体对象的期望状态转换为世界坐标系。
            '''
            # reset root state
            # 我们使用 assets.RigidObject.data.default_root_state 属性获取生成的刚性物体对象的默认根状态。
            # 该默认状态可以从 assets.RigidObjectCfg.init_state 属性中配置，我们在本教程中将其保留为单位状态。
            root_state = cone_object.data.default_root_state.clone()
            # sample a random position on a cylinder around the origins
            # 我们使用 assets.RigidObject.num_instances 属性获取生成的刚性物体对象的数量。
            # 然后，我们使用 omni.isaac.lab.utils.math.sample_cylinder() 函数在原点周围采样随机位置。
            # 我们将采样的位置添加到原点，并将其添加到根状态的位置中。
            root_state[:, :3] += origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 2.5), size=cone_object.num_instances, device=cone_object.device
            )
            # write root state to simulation
            # 我们使用 assets.RigidObject.write_root_pose_to_sim() 和 assets.RigidObject.write_root_velocity_to_sim() 方法将根状态写入模拟。
            cone_object.write_root_pose_to_sim(root_state[:, :7])
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            # 最后，我们使用 assets.RigidObject.reset() 方法重置缓冲区。
            # 这些缓冲区存储了与生成的刚性物体对象的物理交互相关的信息。
            cone_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        # 在推进模拟之前，我们执行 assets.RigidObject.write_data_to_sim() 方法。
        # 此方法将其他数据，例如外部力，写入模拟缓冲区。
        # 在本教程中，我们没有对刚性物体施加任何外部力，因此此方法不是必需的。但是，为了完整性，我们还是加入了它。
        cone_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cone_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_state_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=(1.5, 0.0, 1.0), target=(0.0, 0.0, 0.0))
    # Design scene
    scene_entities, scene_origins = design_scene()
    # 将scene_origins转换为符合sim.device的pytorch张量类型
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
