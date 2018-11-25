def preprocess_embeddings(embeddings, num_feats, T):
    target_len = T * num_feats
    processed_embeddings = []
    for emb in embeddings:
        emb = emb.reshape(-1)
        processed_emb = unify_embedding_length(emb, target_len)
        processed_emb = processed_emb.reshape(-1, num_feats)
        processed_embeddings.append(processed_emb)
        
    return processed_embeddings

# unify feat size to ensure all embeddings are 1024xT
# if embedding is smaller augment it with zeros at the end
# if embedding is larger crop the extra rows
def unify_embedding_length(emb, target_len):
    emb_len = len(emb)
    if emb_len < target_len:
        len_diff = target_len - emb_len
        zero_padding = np.zeros([len_diff])
        return np.hstack((emb, zero_padding))
    elif emb_len > target_len:
        return emb[0:target_len]
    else:
        return emb
    

