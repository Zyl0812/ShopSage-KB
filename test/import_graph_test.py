#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入流程图最小联调测试
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.processor.import_process.main_graph import create_import_graph
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode


def test_import_graph_with_mocked_pdf_conversion():
    """
    使用 mock 方式跳过真实 MinerU 调用，验证 graph 链路和状态更新。
    """
    original_execute = PdfToMdNode._execute_mineru
    original_output_path = PdfToMdNode._get_output_path

    def _mock_execute_mineru(
        self, import_file_path_obj: Path, file_dir_obj: Path
    ) -> int:
        return 0

    def _mock_get_output_path(self, import_file_path: Path, file_dir_path: Path) -> str:
        return str(
            file_dir_path
            / import_file_path.stem
            / "hybrid_auto"
            / f"{import_file_path.stem}.md"
        )

    PdfToMdNode._execute_mineru = _mock_execute_mineru
    PdfToMdNode._get_output_path = _mock_get_output_path

    try:
        graph = create_import_graph()
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%mock file")

            init_state = {
                "file_dir": tmp_dir,
                "import_file_path": str(pdf_path),
            }
            result = graph.invoke(init_state)

        assert result["is_pdf_read_enabled"] is True
        assert result["pdf_path"] == str(pdf_path)
        assert result["md_path"].endswith(r"demo\hybrid_auto\demo.md")
        print("import_graph_test: PASS")
    finally:
        PdfToMdNode._execute_mineru = original_execute
        PdfToMdNode._get_output_path = original_output_path


if __name__ == "__main__":
    test_import_graph_with_mocked_pdf_conversion()
