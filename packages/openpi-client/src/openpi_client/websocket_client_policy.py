import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """通过 WebSocket 与远端策略服务器交互的客户端实现。

    - 该类实现了 BasePolicy 接口，负责：
      1) 与策略服务器建立 WebSocket 连接；
      2) 将本地观测 obs 序列化并发送；
      3) 接收服务器返回的推理结果并反序列化。
    - 服务器端对应实现参见 `WebsocketPolicyServer`。
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """构造函数

        Args:
            host: 策略服务器主机地址（支持直接传 ws:// 或 wss:// URI）。
            port: 策略服务器端口（None 表示使用服务器默认端口）。
            api_key: 可选的鉴权密钥，将以 `Authorization: Api-Key <key>` 发送。
        """
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        # 使用支持 numpy 的 msgpack 打包器，确保高效传输 ndarray 等数据
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        # 等待服务器就绪并建立连接，同时获取服务器元数据
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """返回服务器端在握手阶段发送的 metadata（模型信息/版本等）。"""
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """循环等待服务器就绪并建立连接。

        - 若连接被拒绝(服务器未启动或未就绪)，每 5 秒重试一次。
        - 连接成功后，期望接收首个二进制消息作为 metadata。
        """
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                # 连接建立后，服务器会先发送 metadata（msgpack 格式的 bytes）
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """发送观测并同步等待推理结果。

        Args:
            obs: 输入观测字典（包含图像/状态/文本等，需与服务器端策略期望的格式一致）。

        Returns:
            服务器返回的推理结果字典（例如动作/诊断信息等）。

        Raises:
            RuntimeError: 若服务器返回字符串（通常为错误信息），此处抛出异常并附带原始信息。
        """
        # 将 Python/numpy 结构打包为 msgpack 二进制
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # 期望收到 bytes；若为 str，多为服务器端报错信息
            raise RuntimeError(f"Error in inference server:\n{response}")
        # 解包为 Python 结构（含 numpy 数组）
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        """可选的会话重置接口（当前无状态，占位实现）。"""
        pass
