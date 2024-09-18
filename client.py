import asyncio

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from viam.components.camera import Camera
from viam.services.vision import VisionClient
from viam.utils import dict_to_struct, ValueTypes, Mapping
from collections.abc import Mapping


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key="s4byutclmniiqa5dzok21oftvlw3bcab",
        api_key_id="e74075f5-9193-489f-89b0-d1a951092f89",
    )
    return await RobotClient.at_address("mac-server-main.wszwqu7wcv.viam.cloud", opts)


async def main():
    machine = await connect()

    print("Resources:")
    print(machine.resource_names)

    vision_object_tracker = VisionClient.from_robot(machine, "vision-object-tracker")
    dets = await vision_object_tracker.get_detections_from_camera(
        camera_name="camera-1"
    )
    old_label = dets[0].class_name
    # do_command_input = {"relabel": [[old_label, "robin"]]}
    do_command_input = {
        "relabel": {"person_2_20240916_140915": "nicolas", old_label: "robin"}
    }
    do_command_res = await vision_object_tracker.do_command(do_command_input)
    print(do_command_res)
    dets = await vision_object_tracker.get_detections_from_camera(
        camera_name="camera-1"
    )
    new_label = dets[0].class_name
    print(new_label)
    await asyncio.sleep(3)
    dets = await vision_object_tracker.get_detections_from_camera(
        camera_name="camera-1"
    )
    label = dets[0].class_name
    print(label)
    # vision_object_tracker_return_value = await vision_object_tracker.do_command(
    #     do_command_input
    # )
    # print(
    #     f"vision-object-tracker get_properties return value: {vision_object_tracker_return_value}"
    # )

    # Don't forget to close the machine when you're done!
    await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
