import os.path
import uvicorn

from fastapi import FastAPI, File, UploadFile, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from schema.upload_schema import UploadResponse
from schema.task_schema import TaskStatusResponse
from core.paths import get_front_page_dir
from core.deps import get_task_service
from core.deps import get_import_file_service
from service.import_file_service import ImportFileService
from service.task_service import TaskService
from processor.import_process.base import setup_logging


def create_app() -> FastAPI:
    '''
    创建FastAPI实例
    '''
    # 1. 实例化FastAPI实例
    app = FastAPI(description='知识库导入')
    
    # 2. 跨域配置（顺便配上:当前项目不会出现）---浏览器会出现
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许任意的源
        allow_credentials=True,  # 允许cookie中携带任意的自定义参数
        allow_methods=["*"],  # 允许任意的请求方式
        allow_headers=["*"],  # 允许请求头中携带任意的我自定义参数
    )
    
    # 3. 配置静态文件目录
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
        
    # 4. 注册路由
    register_routes(app)
    
    # 5. 返回实例
    return app


def register_routes(app: FastAPI):
    '''
    注册路由
    '''
    # 1. 处理导入页面访问请求
    @app.get('/import')
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), 'import.html'))
    
    # 2. 上传请求
    @app.post('/upload', response_model=UploadResponse)
    async def upload_file_endpoint(background_tasks: BackgroundTasks, file: UploadFile=File(...), service: ImportFileService = Depends(get_import_file_service)):
        # 2.1 上传文件（本地/MinIO）
        task_id, file_dir, import_file_path = service.process_upload_file(file)
        # 2.2 运行后台任务（跑graph的整个流程）
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        
        # 2.3 返回
        return UploadResponse(message='文件上传成功', task_id=task_id)
    
    # 3. 查询任务状态请求
    @app.get('/status/{task_id}', response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id: str, task_service: TaskService = Depends(get_task_service)):
        '''根据任务ID查询任务状态'''
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)
        
    
    
    
if __name__ == "__main__":
    '''
    启动web服务器
    '''
    setup_logging()
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
