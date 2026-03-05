import base64
import logging
import os
import re

from pathlib import Path

from typing import Dict, List, Tuple

from openai import OpenAI

from processor.import_process.base import BaseNode
from processor.import_process.config import get_config
from processor.import_process.exceptions import FileProcessingError, ImageProcessingError, ValidationError
from processor.import_process.state import ImportGraphState
from utils.minio_util import get_minio_client


class MdImgNode(BaseNode[ImportGraphState]):
    """
    处理markdown图片节点类
    """

    name = "MdImgNode"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        Args:
            state:上一个节点处理之后的state最新状态

        Returns:
            state:处理之后的state最新状态（md_content = process_md）
        """
        config = get_config()
        # 1. 处理文件路径(md内容、md的path、图片目录)
        md_content, md_path_obj, img_dir = self._get_img_md_content(state)

        if not img_dir.exists():
            # 图片不用处理，直接更新state的md_content
            self.logger.warning(f"文件{md_path_obj.name}暂无图片要处理")
            state["md_content"] = md_content
            return state

        # 2. 扫描并处理图片
        target_images_context = self._scan_and_context(img_dir, md_content, config)

        # 3. 用VLM为图片生成摘要
        img_summaries = self._extract_img_summary(
            md_path_obj.stem, target_images_context, config
        )

        # 4. 图片上传到MinIO，将图片描述和图片地址回写到md_content中
        # 4.1 本地图片上传到minio--->remote_url
        # 4.2 替换md中的图片的本地url，以及vlm生成的摘要
        new_md_content = self._upload_img_and_update_md(
            config, md_path_obj.name, md_content, img_summaries, target_images_context
        )
        
        # 5. 将更新后的内容备份
        self._backup_new_md_file(md_path_obj, new_md_content)
        
        # 5. 返回state
        state["md_content"] = new_md_content
        return state
    

    def _get_img_md_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        Args:
            state:上一个节点处理之后的state最新状态

        Returns:
            md_content:处理之后的md内容
            md_path:md的路径
            img_dir:图片的目录
        """
        self.log_step("step1", "读取md内容以及构建图片目录")

        # 1. 从state中取path
        md_path = state.get("md_path", "")

        # 2. 判断路径是否有内容
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        # 3. Path标准化
        md_path_obj = Path(md_path)

        # 4. 判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径不可用", self.name)

        # 5. 获取md内容
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 6. 获取图片目录
        img_dir = md_path_obj.parent / "images"

        # 7. 返回md内容、md路径和图片目录
        return md_content, md_path_obj, img_dir

    def _scan_and_context(
        self, img_dir: Path, md_content: str, config
    ) -> List[Tuple[str, str, Tuple[str, str, str]]]:
        """
        扫描处理图片并返回所有有效图片的丰富信息（1. img_name, 2. img_path, 3. 图片的上下文）
        图片的上下文策略：
        1. max_token：固定token数量截取上下文    缺点：丢失语义
        2. hybrid（混合策略）：
            a. 先找到当前图片的最近标题，拿到该标题的内容；
            b. 从图片开始找内容一直到标题处；
            c. 根据开始索引和结束索引定位到两个索引间的内容；
            d. 利用段落和max_token选择最终留下的内容

        Args:
            img_dir:图片的目录
            md_content:md内容
            config:配置文件
        Returns:
            List[Tuple[str, str, Tuple[str, str, str]]]: 所有有效图片的丰富信息（1. img_name, 2. img_path, 3. 图片的上下文(标题， 上文， 下文))
        """
        self.log_step("step2", f"扫描图片文件目录{img_dir}")
        target_images_context = []

        # 1. 遍历图片文件目录
        for file_name in os.listdir(img_dir):
            # 1.1 获取文件后缀
            file_type = os.path.splitext(file_name)[1]

            # 1.2 如果该文件不是有效图片文件，跳过
            if file_type.lower() not in config.image_extensions:
                continue

            # 1.3 构建img_path
            img_path = str(img_dir / file_name)

            # 1.4 构建图片的上下文
            img_context = self._extract_context_with_limit(
                md_content, file_name, config.img_content_length
            )

            # 1.5 如果该图片没有上下文，跳过
            if not img_context:
                self.logger.warning("md文件中暂未提取到可用的图片")
                continue

            # 1.6 提取到当前图片的唯一上下文内容（为方便使用获取第一个）
            primary_img_context = img_context[0]

            # 1.7 存储到列表中
            target_images_context.append((file_name, img_path, primary_img_context))

        self.logger.info(f"找到了{len(target_images_context)}个图片")
        return target_images_context

    def _extract_context_with_limit(
        self, md_content: str, file_name: str, max_chars=200
    ) -> List[Tuple[str, str, str]]:
        """
        从md文档中提取图片上下文信息
        使用正则查找图片在md的位置
        Args:
            md_content: 要操作的md
            file_name: 要操作的图片的文件名
            max_chars: 上下文的最大字符数

        Returns:
            List[Tuple[str, str, str]]: 离图片最近的一个标题，上文， 下文
        """
        # 1. 定义正则的匹配模式（从md中找到图片）,标准图片的语法结构：![图片的描述](图片路径"提示")
        """
        正则规则解释：
            r: python不要再对正则中的字符做转义了
            !: md语法
            []: 需要正则转义，在正则中表示字符集(a-z A-Z 0-9 + /)
            .: 任意字符
            *: 任意字符出现的数量0个或多个
            ?: 非贪婪模式
            (): 需要正则转义，在正则中表示捕获组
            re.escape(file_name): 需要正则转义，防止文件名中出现特殊字符(.)
        """
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(file_name) + r".*?\)")

        # 2. 从md中先定位图片位置（先按行切分，在逐行匹配）
        lines = md_content.split("\n")
        img_context = []
        for index, line in enumerate(lines):
            if not pattern.search(line):
                # 不是图片
                continue  # 继续下一行

            # 2.1 找到图片后，定位这张图片最近上文的标题内容和索引
            head_title = ""
            head_index = -1
            for i in range(index - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", lines[i]):
                    head_title = lines[i]
                    head_index = i
                    break

            # 2.2 定义要截取上文的索引
            pre_content_start_index = head_index + 1

            # 2.3 提取上文内容（自下而上）
            pre_content = lines[pre_content_start_index:index]
            img_pre_context = self._extract_img_context(
                pre_content, max_chars, direction="front"
            )

            # 2.3 提取下文内容（自上而下）
            section_index = len(lines)
            for i in range(index + 1, len(lines)):
                if re.match(r"^#{1,6}\s+", lines[i]):
                    section_index = i
                    break

            post_content = lines[index + 1 : section_index]
            img_post_context = self._extract_img_context(
                post_content, max_chars, direction="back"
            )

            img_context.append(
                (head_title, img_pre_context, img_post_context)
            )  # 只会有一个三元组对象，除非这个图片在md中有多处引用

        return img_context

    def _extract_img_context(
        self, content: List[str], max_chars: int, direction: str
    ) -> str:
        """
        提取图片到上下标题之间的内容（段落）
        策略：markdown中的段落按\n分割，每一行后面都有两个空格
        Args:
            content (List[str]): 图片上下文内容
            max_chars (int): 最大字符数
            direction (str): 方向，front（自下而上）或back（自上而下）

        Returns:
            str: 提取的上下文内容
        """
        current_paramgraph = []  # 存储当前遍历到的内容
        final_paragraphs = []  # 存储最终提取到的段落

        # 1. 遍历每一行
        for line in content:
            clean_strip = line.strip()
            if not clean_strip:
                # 没有内容
                if current_paramgraph:
                    final_paragraphs.append("\n".join(current_paramgraph))
                    current_paramgraph = []
            else:
                # 有内容
                if re.match(r"!\[.*?\]\(.*?\)", clean_strip):
                    # 图片引用行，不计入上下文，但作为一个段落分隔符
                    if current_paramgraph:
                        final_paragraphs.append("\n".join(current_paramgraph))
                        current_paramgraph = []
                    continue

                current_paramgraph.append(line)

        if current_paramgraph:
            final_paragraphs.append("\n".join(current_paramgraph))

        # 2. 判断方向
        if direction == "front":
            # 上文：从下往上取，拿最近的段落
            final_paragraphs.reverse()

        # 3. 判断final中收集到的段落字符长度是否超过了max_chars
        total = 0
        selected_paragraphs = []
        for para in final_paragraphs:
            if total + len(para) > max_chars and selected_paragraphs:
                break
            selected_paragraphs.append(para)
            total += len(para)

        # 4. 在判断一次方向，返回selected_paragraphs
        if direction == "front":
            # 从下往上取完段落后还需调转方向符合语义
            selected_paragraphs.reverse()

        return "\n\n".join(selected_paragraphs)

    def _extract_img_summary(
        self,
        file_title: str,
        target_images_context: List[Tuple[str, str, Tuple[str, str, str]]],
        config,
    ) -> Dict:
        """
        为所有图片生成摘要（VLM）
        Args:
            file_title (str): 文件标题
            target_images_context (List): 图片上下文信息
            config
        Returns:
            Dict: {'图片名字'：'图片摘要'}
        """
        self.log_step("step3", "上传图片并更新md的摘要")
        config = get_config()
        summaries = {}
        # 1. 构建openai客户端
        try:
            client = OpenAI(
                api_key=config.openai_api_key, base_url=config.openai_api_base
            )
        except Exception:
            logging.error("OpenAI VLM 客户端创建失败")
            return summaries

        # 2. 发送请求（提取摘要）
        for img_name, img_path, img_context in target_images_context:
            # TODO:每分钟最大请求次数限制

            summary = self._get_img_summary(
                config, client, file_title, img_path, img_context
            )
            summaries[img_name] = summary

        logging.info(f"生成{len(summaries)}个图片摘要")
        # 4. 返回映射表
        return summaries

    def _get_img_summary(
        self, config, client, file_title: str, img_path: str, img_context: Tuple
    ):
        """
        为图片生成摘要（VLM）
        Args:
            config (Config): 配置信息
            client (OpenAI): openai客户端
            file_title (str): 文件标题
            img_path (str): 图片路径
            img_context (Tuple): 图片上下文信息

        Returns:
            str: 图片摘要
        """
        # 1. 解包 img_context 构建上下文
        section_title, pre_context, post_context = img_context

        # 2. 判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_context:
            context_parts.append(post_context)

        # 3. 构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无上下文可用"

        # 4. 读取图片文件
        with open(img_path, "rb") as f:
            local_img_content = base64.b64encode(f.read()).decode("utf-8")

        # 5. 调用VLM发送请求
        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                                背景信息：
                                    1. 所属文档标题："{file_title}"
                                    2. 图片上下文：{final_context}
                                    请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                    生成一个精准的中文标题（不要包含"图片"二字）。""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                },
                            },
                        ],
                    }
                ],
            )
            summary = response.choices[0].message.content.strip().replace("\n", " ")
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path}: {e}")
            return "图片描述"

    def _upload_img_and_update_md(
        self, config, file_name, md_content, img_summaries, target_images_context
    ):
        """
        上传图片到minio以及替换md中的图片url和摘要
        Args:
            md_content (str): markdown内容
            img_summaries (dict): 图片摘要映射表
            target_images_context (list): 图片上下文列表
        Returns:

        """
        self.log_step("step4", "上传图片并更新md的摘要")
        remote_urls = {}

        # 1. 构建minio客户端
        minio_client = get_minio_client()

        if minio_client is None:
            self.logger.warning("无法将本地图片上传到MinIO，跳过图片URL替换")
            return md_content

        # 2. 遍历图片信息列表
        for img_name, img_path, _ in target_images_context:
            # 2.1 构建对象名字
            object_name = f"{file_name}/{img_name}"

            # 2.2 开始上传
            try:
                minio_client.fput_object(
                    bucket_name=config.minio_bucket,
                    object_name=object_name,
                    file_path=img_path,
                )
                # 2.3 手动拼接远程地址
                remote_url = (
                    config.get_minio_base_url()
                    + "/"
                    + config.minio_bucket
                    + "/"
                    + object_name
                )
                self.logger.info(f"图片{img_name}上传到MinIO成功: {remote_url}")
                remote_urls[img_name] = remote_url

            except Exception as e:
                self.logger.warning(f"图片{img_name}上传到MinIO失败: {e}")
                continue

        self.logger.info(f"成功上传{len(remote_urls)}个图片到MinIO")

        # 3. 替换图片中的摘要和地址到md中
        new_md_content = md_content

        for img_name, img_summary in img_summaries.items():
            # 3.1 提取远程地址
            remote_url = remote_urls.get(img_name)

            if not remote_url:
                self.logger.warning(f"图片{img_name}未上传到MinIO")
                continue

            # 3.2 替换url和摘要
            replace_pattern = re.compile(
                r"!\[[^\]]*\]\([^)\n]*" + re.escape(img_name) + r"[^)\n]*\)",
                re.IGNORECASE,
            )
            new_md_content = replace_pattern.sub(
                f"![{img_summary}]({remote_url})", new_md_content
            )

        return new_md_content
    
    def _backup_new_md_file(self, md_path_obj: Path, new_md_content: str) -> str:
        self.log_step("step_5", "备份新文件")
        
        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)

        return str(new_file_path) 

if __name__ == "__main__":
    img_md_node = MdImgNode()

    state = ImportGraphState(
        {
            "md_path": r"D:\atguigu\shopkeer_brain\knowledge\processor\import_process\output_temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"
        }
    )

    img_md_node.process(state)