from annoy import AnnoyIndex
import pandas as pd
import pdb
from model import load_image
import time


# bikin pred model untuk extract feature vector
from fastai.vision import load_learner, open_image


class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()
        
def get_vector(img, model): 
    sf = SaveFeatures(model.model[1][4]) # layer ke-empat sebelum softmax (_fc)
    model.
    _ = model.predict(open_image(img))
    vectors = sf.features
    sf.remove()
    return vectors[0]

def get_similar_image_by_vector(img_vector, df, t): 
    """ df nya itu isi image_path dengan index dari annoy yang bener """
    start = time.time()
    similar_img_ids = t.get_nns_by_vector(img_vector, 8)
    
    end = time.time() - start
    print(end, 's')
    return df.iloc[similar_img_ids]

if __name__== "__main__": 

    learn = load_learner('./dataset')
    df_new = pd.read_pickle('./dataset/models/image_Resnet50_vectors.pkl')
    f = 512 # dimension vector image nya (resnet 50 pake 512)
    u = AnnoyIndex(f, 'euclidean')
    u.load('./baseline_annoy.ann')

    # random_search = u.get_nns_by_item(2,3)
    random_image = 'dataset/test/004df33b3487b4691f638c5603341901.jpg'
    vector_ri = get_vector(random_image, learn)
    similar_ri = get_similar_image_by_vector(vector_ri, df_new, u)
    pdb.set_trace()

# get tree_index to image_index

# show image -> ini gimana coba caranya buat display di terminal anjir (libsixel kalo gak salah bisa deh tapi harus ada sudo)