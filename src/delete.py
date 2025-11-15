from src.google_drive import get_file_by_uuid, delete_file_by_id
import numpy as np

def remove_by_uuid(faiss_index, drive_service, uuid):
    file = get_file_by_uuid(drive_service, uuid)
    if not file:
        return 
    upload_id = file['id']

    emb_metadata_row = faiss_index.emb_metadata[faiss_index.emb_metadata.img_path == upload_id]
    image_id = emb_metadata_row.squeeze()['id']
    image_index = emb_metadata_row.index[0]

    # Remove from the embedding metadata
    try:
        faiss_index.emb_metadata.drop(image_index, inplace=True)
        print("Metadata deleted.")
    except Exception as e:
        print(f"Couldn't delete the metadata from the FAISS index metadata file. Reason: {e}")

    # Remove from faiss
    # Fetch the embedding vector first to rollback if there is an error
    emb_vector = faiss_index.index.index.reconstruct(int(image_id)).reshape(1, -1)
    try:
        faiss_index.index.remove_ids(np.array([image_id], dtype='int64'))
        print("Embedding removed.")
    except Exception as e:
        print(f"Can't remove the image embeddings from the FAISS index. Reason: {e}")
        print(f"Rolling back the metadata removal...")
        faiss_index.emb_metadata.loc[image_index] = emb_metadata_row.iloc[0]

    # Remove from the drive
    try:
        deleted = delete_file_by_id(drive_service, upload_id)
    except Exception as e:
        print(f"Can't delete the image from the drive. Reason: {e}")
        print("Rolling back the metadata and embedding removal...")
        faiss_index.emb_metadata.loc[image_index] = emb_metadata_row.iloc[0]
        faiss_index.index.add_with_ids(emb_vector, np.array([image_id], dtype=np.int64))

    # Save the changes
    faiss_index._save() 
    print("\033[1mDeleting Done.\033[0m")