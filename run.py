from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1", map_location="mps")

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
result = model.extract_entities(
    text,
    ["company", "person", "product", "location"],
    include_confidence=True,
    include_spans=True,
)
print(result)
