import logging
from io import BytesIO
from pathlib import Path

import grpc
from PIL import Image

from .vehicle_pose_detection_service_pb2_grpc import VehiclePoseDetectionStub
from .vehicle_pose_detection_service_pb2 import (
    Empty,
    StartSessionResponse, 
    DetectRequest, 
    DetectResponse,
    Detection,
    DetectionClass,
    Box2D,
    Box3D,
    Box3DProjection,
    StopSessionRequest)

from ..viz import visualization
from ..detection_info import DetectionInfo, get_camera_intrinsics

class UnexpectedResponseType(Exception):
    pass

class VehiclePoseDetectionClient():
    def __init__(self, ip_address: str, port: str, ping_timeout_in_seconds: float):
        self.ping_timeout_in_seconds = ping_timeout_in_seconds
        
        self.channel = grpc.insecure_channel(
            target=f"{ip_address}:{port}", 
            options=[
                ("grpc.max_send_message_length", 10_000_000),
                ("grpc.max_receive_message_length", 10_000_000),
                ("grpc.max_message_length", 10_000_000)
            ])    

        self.stub = VehiclePoseDetectionStub(self.channel)
        self.Ping()

    def Ping(self):
        request = Empty()

        logging.getLogger(__name__).info("Sending Ping to server with timeout %s ...)", self.ping_timeout_in_seconds)

        response = self.stub.Ping(
            request, 
            wait_for_ready=True, 
            timeout=self.ping_timeout_in_seconds)

        if not isinstance(response, Empty):
            raise UnexpectedResponseType()

    
    def StartSession(self) -> int:
        request = Empty()
        response: StartSessionResponse = self.stub.StartSession(request)
        return response.session_id

    def Detect(self, session_id: int, frame_id: int, frame_image: Image.Image):
        frame_bytesio = BytesIO()
        frame_image.save(frame_bytesio, format="jpeg")
        frame_bytes = frame_bytesio.getvalue()

        request = DetectRequest(
            session_id=session_id,
            frame_id=frame_id,
            frame=frame_bytes)
        
        response: DetectResponse = self.stub.Detect(request)

        session_id_resp, frame_id_resp, detections = \
            response.session_id, response.frame_id, response.detections

        assert session_id == session_id_resp
        assert frame_id == frame_id_resp

        detection_infos = list()
        for detection in detections:
            print(detection.detection_class)
            detection_infos.append(
                DetectionInfo(
                    detection.detection_class,
                    
                    None,
                    
                    detection.box_2D.xmin,
                    detection.box_2D.ymin,
                    detection.box_2D.xmax,
                    detection.box_2D.ymax,

                    detection.box_3D.dimension_h,
                    detection.box_3D.dimension_w,
                    detection.box_3D.dimension_l,

                    detection.box_3D.location_x,
                    detection.box_3D.location_y,
                    detection.box_3D.location_z,

                    detection.box_3D.rotation,
                    
                    detection.box_3D.score))

        viz_bytes, _ = visualization(frame_image, get_camera_intrinsics(), detection_infos)
            
        viz_bytesio = BytesIO(viz_bytes)
            
        viz_image = Image.open(viz_bytesio)
        viz_image.show()

        return detection_infos

    def StopSession(self, session_id: int) -> None:
        request = StopSessionRequest(
            session_id=session_id)

        response = self.stub.StopSession(request)
        
        if not isinstance(response, Empty):
            raise UnexpectedResponseType()


vehicle_pose_detection_client = VehiclePoseDetectionClient(
    "3.123.206.99",
    "50052",
    60
)

_session_id = vehicle_pose_detection_client.StartSession()

_frame_image = Image.open(BytesIO(Path("kitti/testing/image_2/000015.png").read_bytes()))
_frame_image = Image.open(BytesIO(Path("input.jpg").read_bytes()))
_frame_image.show()
_detection_infos = vehicle_pose_detection_client.Detect(_session_id, 0, _frame_image)

vehicle_pose_detection_client.StopSession(_session_id)
