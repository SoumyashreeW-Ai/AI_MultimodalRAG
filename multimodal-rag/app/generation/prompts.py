"""Grounded RAG prompt templates."""
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "Do not invent facts or use outside knowledge. Cite sources where appropriate. "
    "Distinguish between information derived from text and from images. If an image is provided, "
    "explicitly state when you are describing visual information. If the answer cannot be found "
    "in the supplied context, say so clearly. Keep answers concise and list source references. "
    "Answer in 1-2 short sentences whenever possible. For simple list requests, provide up to 5 items "
    "separated by commas. Avoid reproducing long tables, full documents, or image-instruction text."
)

def build_user_context(query: str, text_blocks: list[str], image_blocks: list[dict]) -> str:
    parts = [f"USER QUESTION:\n{query}", "", "TEXT SOURCES:"]
    for i, t in enumerate(text_blocks, start=1):
        parts.append(f"Source {i}:\n{t}\n")

    parts.append("IMAGE SOURCES:")
    for i, img in enumerate(image_blocks, start=1):
        parts.append(f"Image {i}:\nfilename: {img.get('filename')}\npage: {img.get('page')}\nimage description: {img.get('description') or '[no description]'}\nimage: {img.get('image_tag') or '[image omitted]'}\n")

    parts.append("Please answer concisely and cite which sources you used. If information is missing, say that it was not found in the provided context.")
    return "\n".join(parts)
