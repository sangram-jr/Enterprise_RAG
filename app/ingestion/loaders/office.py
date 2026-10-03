# import logfire
# from unstructured.partition.auto import partition

# def parse_office(file_path: str):
#     """
#     Parses Office documents (.docx, .pptx) using the Unstructured library.
#     Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
#     """
#     with logfire.span("📄 Office Document Parsing", filename=file_path):
#         try:
#             # Unstructured automatically detects if it's docx or pptx
#             elements = partition(filename=file_path)
#             full_text = "\n".join([str(el) for el in elements])
            
#             if not full_text.strip():
#                 logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
#             else:
#                 logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

#             return full_text
#         except Exception as e:
#             logfire.error(f"❌ Office Parse Failed: {e}")
#             raise e


import logfire


def parse_office(file_path: str):
    """
    Parse DOCX/PPTX files using Unstructured's explicit loaders.
    """

    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            ext = file_path.lower().rsplit(".", 1)[-1]

            if ext == "pptx":
                from unstructured.partition.pptx import partition_pptx
                elements = partition_pptx(filename=file_path)

            elif ext == "docx":
                from unstructured.partition.docx import partition_docx
                elements = partition_docx(filename=file_path)

            else:
                raise ValueError(f"Unsupported office file type: {ext}")

            full_text = "\n".join(str(el) for el in elements)

            if not full_text.strip():
                logfire.warning(
                    f"Unstructured returned empty text for {file_path}"
                )
            else:
                logfire.info(
                    f"Successfully parsed {len(full_text)} characters"
                )

            return full_text

        except Exception as e:
            logfire.error(f"Office Parse Failed: {e}")
            raise