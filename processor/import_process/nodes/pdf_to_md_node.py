import subprocess
from pathlib import Path
from time import time
from typing import Tuple

from processor.import_process.base import BaseNode

# from processor.import_process.base import setup_logging
from processor.import_process.exceptions import (
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode[ImportGraphState]):
    """
    PDF转换MD节点
    """

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args：
            state (ImportGraphState):
        Returns:
            ImportGraphState:
        """
        # 1. 对参数校验，获取输入文件和所在路径
        import_file_path, file_dir_path = self._validate_paths(state)

        # 2. 利用MinerU工具解析pdf成为md
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)
        self.logger.warning("已跳过MinerU解析步骤（临时联调）")

        # 3. 获取md的path
        md_path = self._get_output_path(import_file_path, file_dir_path)

        # 4. 更新state，字典的md_path
        state["md_path"] = md_path

        # 5. 返回state
        return state

    def _validate_paths(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        验证pdf路径和文件目录是否存在
        Args:
            state: 该节点接收到的状态
        Returns:
            bool: pdf路径和文件目录是否存在
        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入pdf文件的路径
        import_file_path = state.get("import_file_path", "")
        # 2. 获取解析后的输出路径
        file_dir = state.get("file_dir", "")

        # 3. 校验输入的文件是否存在
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        # 4. Path标准化
        import_file_path_obj = Path(import_file_path)

        # 5. 校验是否是一个真实的路径
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)

        # 6. 判断输出目录是否存在
        if not file_dir:
            # 默认目录做兜底
            file_dir = import_file_path_obj.parent

        # 7. 返回输入文件以及输出目录的标准path
        file_dir_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径：{import_file_path}")
        self.logger.info(f"输出文件目录：{file_dir}")

        # 8. 返回输入文件以及输出目录的标准path
        return import_file_path_obj, file_dir_obj

    def _execute_mineru(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        """
        执行mineru转换命令,实时输出日志并输出状态码
        Args:
            import_file_path_obj: 输入文件路径
            file_dir_obj: 输出目录路径
        Returns:
            int: mineru转换命令的输出状态码
        """
        self.log_step("step2", "开始执行MinerU解析pdf命令")
        # 1. 构建命令行
        cmd = [
            "mineru",
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
            "--source",
            "local",
        ]
        process_start_time = time()

        # 2. 执行命令行（子进程执行命令行），自动读取到主进程的环境变量
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",  # 替换乱码文字
            text=True,  # 保证输出内容是字符串
            encoding="utf-8",
            bufsize=1,  # 设置按行缓冲区大小为1，实现实时输出
        )

        # 3. 获取日志信息
        assert proc.stdout is not None
        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line.strip()}")

        # 4. 等待子进程结束
        processed_code = proc.wait()
        process_end_time = time()
        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path_obj.name},耗时：{process_end_time - process_start_time:.2f}秒"
            )
        else:
            self.logger.error(f"MinerU解析PDF失败，状态码：{processed_code}")

        # 5. 返回状态码
        return processed_code

    def _get_output_path(self, import_file_path: Path, file_dir_path: Path) -> str:
        """
        计算输出路径，根据MinerU输出结构构建md文件路径
        Args:
            import_file_path: PDF文件路径
            file_dir_path: 文件目录路径
        Returns:
            Path: md文件路径
        """
        # stem: 文件名，suffix: 文件后缀
        file_name = import_file_path.stem
        output_path = file_dir_path / file_name / "hybrid_auto" / f"{file_name}.md"
        return str(output_path)


# if __name__ == "__main__":
#     setup_logging()
#     pdf_to_md_node = PdfToMdNode()
#     init_state = {
#         "import_file_path": r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf",
#         "file_dir": r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\output_temp_dir",
#     }
#     pdf_to_md_node.process(init_state)
