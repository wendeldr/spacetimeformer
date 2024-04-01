import numpy as np


with open("//media/dan/Data/git/spacetimeformer/enc_x.npy", "rb") as f:
    enc_x = np.load(f)

with open("//media/dan/Data/git/spacetimeformer/enc_y.npy", "rb") as f:
    enc_y = np.load(f)

with open("//media/dan/Data/git/spacetimeformer/dec_x.npy", "rb") as f:
    dec_x = np.load(f)

with open("//media/dan/Data/git/spacetimeformer/dec_y.npy", "rb") as f:
    dec_y = np.load(f)

# ===
with open("//media/dan/Data/git/spacetimeformer/enc_x1.npy", "rb") as f:
    enc_x1 = np.load(f)

with open("//media/dan/Data/git/spacetimeformer/enc_y1.npy", "rb") as f:
    enc_y1 = np.load(f)

with open("//media/dan/Data/git/spacetimeformer/dec_x1.npy", "rb") as f:
    dec_x1 = np.load(f)

with open("//media/dan/Data/git/spacetimeformer/dec_y1.npy", "rb") as f:
    dec_y1 = np.load(f)




print(f"enc_x same: {np.all(enc_x == enc_x1)}")
print(f"enc_y same: {np.all(enc_y == enc_y1)}")
print(f"dec_x same: {np.all(dec_x == dec_x1)}")
print(f"dec_y same: {np.all(dec_y == dec_y1)}")
print("-------------------")

with open("/media/dan/Data/git/spacetimeformer/enc_space_emb.npy", "rb") as f:
    emb_space = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_space_emb1.npy", "rb") as f:
    emb_space1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_valueTime_emb.npy", "rb") as f:
    emb_time = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_valueTime_emb1.npy", "rb") as f:
    emb_time1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_out.npy", "rb") as f:
    enc_out = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_out1.npy", "rb") as f:
    enc_out1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/x.npy", "rb") as f:
    x = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/x1.npy", "rb") as f:
    x1 = np.load(f)


with open("/media/dan/Data/git/spacetimeformer/dec_vt_emb.npy", "rb") as f:
    dec_vt_emb = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/dec_vt_emb1.npy", "rb") as f:
    dec_vt_emb1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/dec_s_emb.npy", "rb") as f:
    dec_s_emb = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/dec_s_emb1.npy", "rb") as f:
    dec_s_emb1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/forecast.npy", "rb") as f:
    forecast = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/forecast1.npy", "rb") as f:
    forecast1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/recon_out.npy", "rb") as f:
    recon_out = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/recon_out1.npy", "rb") as f:
    recon_out1 = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_var_idxs.npy", "rb") as f:
    enc_var_idxs = np.load(f)

with open("/media/dan/Data/git/spacetimeformer/enc_var_idxs1.npy", "rb") as f:
    enc_var_idxs1 = np.load(f)

print(f"emb_time same: {np.all(emb_time == emb_time1)}")
print(f"emb_space same: {np.all(emb_space == emb_space1)}")
print(f"enc_out same: {np.all(enc_out == enc_out1)}")
print("===================")
print(f"x same: {np.all(x == x1)}")
print("===================")
print(f"dec_vt_emb same: {np.all(dec_vt_emb == dec_vt_emb1)}")
print(f"dec_s_emb same: {np.all(dec_s_emb == dec_s_emb1)}")
print(f"forecast same: {np.all(forecast == forecast1)}")
print(f"recon_out same: {np.all(recon_out == recon_out1)}")
print(f"enc_var_idxs same: {np.all(enc_var_idxs == enc_var_idxs1)}")
print("-------------------")

# ===

with open("//media/dan/Data/git/spacetimeformer/forecast_output.npy", "rb") as f:
    forecast_out = np.load(f)


with open("//media/dan/Data/git/spacetimeformer/forecast_output1.npy", "rb") as f:
    forecast_out1 = np.load(f)

print(f"forecast_out same: {np.all(forecast_out == forecast_out1)}")


import pickle
with open("params.pkl", "rb") as f:
    params_pred = pickle.load(f)

with open("params1.pkl", "rb") as f:
    params_test = pickle.load(f)

# params are list of params for each layer as numpy arrays, check if they are all the same
test=[]
for p1, p2 in zip(params_pred, params_test):
    test.append(np.all(p1 == p2))
print(f"params same: {np.all(test)}")