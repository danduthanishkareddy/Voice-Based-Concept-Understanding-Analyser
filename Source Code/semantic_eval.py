from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(user_text, reference_text):
    user_embedding = model.encode(user_text, convert_to_tensor=True)
    ref_embedding = model.encode(reference_text, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, ref_embedding)
    return round(float(similarity[0][0]) * 100, 2)