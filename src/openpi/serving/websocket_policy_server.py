import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """基于 WebSocket 协议的策略服务器实现。

    - 该服务器负责：
      1) 接收客户端连接并建立 WebSocket 会话；
      2) 在连接建立时发送服务器元数据（模型信息等）；
      3) 循环接收观测数据，调用策略进行推理，并返回动作结果；
      4) 记录并返回推理时间统计信息；
      5) 提供健康检查端点 `/healthz`。
    - 客户端实现参见 `websocket_client_policy.py`。
    - 当前仅实现 `load` 和 `infer` 方法。
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """构造函数

        Args:
            policy: 要服务的策略实例（实现 BasePolicy 接口）。
            host: 服务器监听地址（默认 "0.0.0.0" 表示监听所有网络接口）。
            port: 服务器监听端口（None 表示使用系统分配的随机端口）。
            metadata: 服务器元数据字典，将在连接建立时发送给客户端（如模型版本、配置等）。
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # 设置 WebSocket 服务器日志级别为 INFO
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """启动服务器并永久运行（同步接口）。

        内部调用 asyncio.run() 来运行异步服务器。
        """
        asyncio.run(self.run())

    async def run(self):
        """异步运行 WebSocket 服务器。

        - 创建 WebSocket 服务器实例，绑定到指定主机和端口；
        - 设置无压缩、无大小限制（适合传输大型观测数据）；
        - 注册健康检查处理器；
        - 启动服务器并永久运行。
        """
        async with _server.serve(
            self._handler,  # 连接处理器
            self._host,     # 监听地址
            self._port,     # 监听端口
            compression=None,  # 禁用压缩（减少延迟）
            max_size=None,     # 无消息大小限制
            process_request=_health_check,  # HTTP 请求预处理器（健康检查）
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """处理单个 WebSocket 连接的异步处理器。

        连接处理流程：
        1. 记录客户端连接信息；
        2. 创建 msgpack 打包器（支持 numpy 数组序列化）；
        3. 发送服务器元数据给客户端；
        4. 进入主循环：接收观测 → 推理 → 返回动作 → 记录时间统计。

        Args:
            websocket: WebSocket 服务器连接对象。
        """
        logger.info(f"Connection from {websocket.remote_address} opened")
        # 为每个连接创建独立的 msgpack 打包器
        packer = msgpack_numpy.Packer()

        # 连接建立后，首先发送服务器元数据（模型信息、版本等）
        await websocket.send(packer.pack(self._metadata))

        # 用于记录上一次请求的总处理时间（包含网络传输时间）
        prev_total_time = None
        
        # 主处理循环：持续接收观测并返回推理结果
        while True:
            try:
                # 记录请求开始时间
                start_time = time.monotonic()
                
                # 接收并反序列化观测数据
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # 执行策略推理并记录推理时间
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # 在返回结果中添加服务器端时间统计
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,  # 推理时间（毫秒）
                }
                
                # 如果有上一次的总处理时间，也一并返回（用于客户端性能分析）
                if prev_total_time is not None:
                    # 只能记录上一次的总时间，因为当前请求的发送时间还未完成
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # 序列化并发送推理结果
                await websocket.send(packer.pack(action))
                
                # 记录当前请求的总处理时间（用于下一次请求的时间统计）
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                # 客户端主动关闭连接
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # 发生未预期错误：发送错误堆栈给客户端，然后关闭连接
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """HTTP 健康检查处理器。

    - 当客户端访问 `/healthz` 路径时，返回 HTTP 200 OK 状态；
    - 其他路径继续正常的 WebSocket 处理流程。

    Args:
        connection: WebSocket 服务器连接对象。
        request: HTTP 请求对象。

    Returns:
        健康检查响应（仅对 `/healthz` 路径），其他情况返回 None 继续正常处理。
    """
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # 继续正常的 WebSocket 请求处理
    return None
