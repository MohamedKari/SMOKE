from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
import signal
import sys

import grpc
from PIL import Image

from . import vehicle_pose_detection_service_pb2_grpc

from .vehicle_pose_detection_service_pb2 import (
    Box2D,
    Box3D,
    Point,
    Box3DProjection,
    Empty,
    Detection,
    StartSessionResponse, 
    DetectRequest,
    DetectResponse,
    StopSessionRequest)

from ..viz import visualization
from ..detection_info import DetectionInfo, get_camera_intrinsics
from ..app import SmokeSession


class RpcHandlingException(Exception):
    def __init__(self, rpc_name: str):
        super().__init__()
        self.rpc_name = rpc_name

    def __str__(self):
        error_str = (
            f" \n"
            f"Error during handling RPC with name {self.rpc_name}.\n" 
            f"Error type was: {type(self.__cause__)}\n"
            f"Error message was: {self.__cause__} \n"
        )
            
        logging.error(error_str)
        return error_str

class SmokeServicer(vehicle_pose_detection_service_pb2_grpc.VehiclePoseDetectionServicer):

    def __init__(self):
        super().__init__()

        self.sessions: Dict[int, SmokeSession] = {}
        self.total_session_counter: int = -1
        self.framecounter = -1

    def log_request(self, rpc_name, request, context): # pylint: disable=unused-argument
        logging.getLogger(__name__).debug(
            "Received gRPC request for method %s by peer %s with metadata %s", 
            rpc_name,
            context.peer(),
            context.invocation_metadata())

    def Ping(self, request: Empty, context) -> Empty:
        try:
            # Input
            self.log_request("Ping", request, context)

            # Process

            # Output

            return Empty()
        except Exception as e:
            raise RpcHandlingException("Ping") from e


    def StartSession(self, request: Empty, context) -> StartSessionResponse:
        try:
            # Input
            self.log_request("StartSession", request, context)
            
            # Process
            self.total_session_counter += 1
            session_id = self.total_session_counter

            self.sessions[session_id] = SmokeSession(get_camera_intrinsics())

            # Output
            start_session_response = StartSessionResponse(
                session_id=session_id)

            return start_session_response
        except Exception as e:
            raise RpcHandlingException("StartSession") from e



    def Detect(self, request: DetectRequest, context) -> DetectResponse:
        try:
            # Input
            ## Input Request
            self.framecounter += 1
            self.log_request("Detect", request, context)
            
            session_id: int = request.session_id
            frame_id: int = request.frame_id
            frame: bytes = request.frame

            ## Input Deserialization
            logging.getLogger(__name__).debug("Frame with frame_id %s has length %s", frame_id, len(frame))
            frame_image: Image = Image.open(BytesIO(frame))

            # Process 
            smoke_session: SmokeSession = self.sessions[session_id]
            detection_output = smoke_session.detect(frame_id, frame_image)
            _, detection_output_projection = visualization(
                frame_image, 
                smoke_session.K_3x4,
                [DetectionInfo.from_array(detection) for detection in detection_output])

            # Output
            ## Output Serialization
            logging.getLogger(__name__).debug("Serializing detection_output.")
            
            detections = []
            for i, detection in enumerate(detection_output):
                detection_info = DetectionInfo.from_array(detection)

                detections.append(
                    Detection(
                        detection_class=detection_info.detection_class_string().upper(),
                        box_2D=Box2D(
                            xmin=detection_info.xmin,
                            ymin=detection_info.ymin,
                            xmax=detection_info.xmax,
                            ymax=detection_info.ymax
                        ),
                        box_3D=Box3D(
                            rotation=detection_info.rot_global,
                            location_x=detection_info.tx,
                            location_y=detection_info.ty,
                            location_z=detection_info.tz,
                            dimension_w=detection_info.w,
                            dimension_l=detection_info.l,
                            dimension_h=detection_info.h,
                            score=detection_info.score
                        ),
                        box_3D_projection=Box3DProjection(
                            p1=Point(
                                x=int(detection_output_projection[i][0][0]),
                                y=int(detection_output_projection[i][1][0])),
                            p2=Point(
                                x=int(detection_output_projection[i][0][1]),
                                y=int(detection_output_projection[i][1][1])),
                            p3=Point(
                                x=int(detection_output_projection[i][0][2]),
                                y=int(detection_output_projection[i][1][2])),
                            p4=Point(
                                x=int(detection_output_projection[i][0][3]),
                                y=int(detection_output_projection[i][1][3])),
                            p5=Point(
                                x=int(detection_output_projection[i][0][4]),
                                y=int(detection_output_projection[i][1][4])),
                            p6=Point(
                                x=int(detection_output_projection[i][0][5]),
                                y=int(detection_output_projection[i][1][5])),
                            p7=Point(
                                x=int(detection_output_projection[i][0][6]),
                                y=int(detection_output_projection[i][1][6])),
                            p8=Point(
                                x=int(detection_output_projection[i][0][7]),
                                y=int(detection_output_projection[i][1][7])),
                        )
                    )
                )

                print(detections[-1])


            ## Output Response
            detect_response = DetectResponse(
                session_id=session_id,
                frame_id=frame_id,
                detections=detections)

            return detect_response
        except Exception as e:
            raise RpcHandlingException("Detect") from e


    def StopSession(self, request: StopSessionRequest, context) -> Empty:
        try:
            # Input
            self.log_request("StopSession", request, context)
            
            session_id: int = request.session_id

            # Process

            smoke_session = self.sessions[session_id]
            smoke_session.stop_session()

            del self.sessions[session_id]

            # Output
            stop_session_response = Empty()

            return stop_session_response
        except Exception as e:
            raise RpcHandlingException("StopSession") from e


def register_stop_signal_handler(grpc_server):

    def signal_handler(signalnum, _):
        logging.getLogger(__name__).info("Processing signal %s received...", signalnum)
        grpc_server.stop(None)
        sys.exit("Exiting after cancel request.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def serve():
    server = grpc.server(
        ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_send_message_length", 10_000_000),
            ("grpc.max_receive_message_length", 10_000_000),
            ("grpc.max_message_length", 10_000_000)
        ])

    vehicle_pose_detection_service_pb2_grpc.add_VehiclePoseDetectionServicer_to_server(
        SmokeServicer(),
        server)

    port = 50052
    server.add_insecure_port(f"[::]:{port}")
    register_stop_signal_handler(server)
    server.start()

    logging.getLogger(__name__).info("Serving 3D Vehicle Pose Detection on port %s!", port)
    server.wait_for_termination()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    serve()