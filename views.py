# from typing_extensions import ParamSpecArgs
from django.shortcuts import render
from django.http.response import HttpResponse

from naomi.naomi import NAMOIimputation

import pandas as pd

import json
import os


import torch
def index(request):
    return render(request, 'index.html')


modelDic = {}
def getModel(uid, dataframe = None, window_size = None):
    if uid in modelDic:
        return modelDic[uid]
    modelDic[uid] = NAMOIimputation(dataframe = dataframe, window_size=window_size)
    print(modelDic)
    return modelDic[uid]

def imputeProcess(request):
    uid = request.POST["uid"]
    naomiIMP = getModel(uid)
    naomiIMP.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, naomiIMP.model.parameters()), lr=7e-4)
    naomiIMP.run_epoch(True, naomiIMP.model, naomiIMP.train_data, 10, naomiIMP.optimizer, batch_size = 64, teacher_forcing=True)

    result = naomiIMP.predict_result()
    result = naomiIMP.scaler.inverse_transform(result.reshape(-1,1)).squeeze()
    result = result.tolist()
    label = naomiIMP.df["time"].values.tolist()
    dic = {"label":label, "value": result}
    return HttpResponse(json.dumps(dic), content_type = "application/json")

def imputation(request):
    if len(request.FILES) == 0:
        return
    file = request.FILES["getCSV"]
    window_size = int(request.POST["windowsize"])
    uid = request.POST["uid"]

    df = pd.read_csv(file)
    df = df.fillna("NaN")

    naomiIMP = getModel(uid, df, window_size)
    naomiIMP.imputation(1)

    result_df = naomiIMP.df
    result = naomiIMP.scaler.inverse_transform(result_df["value"].values.reshape(-1,1)).squeeze()
    result = result.tolist()
    label = result_df["time"].values.tolist()
    dic = {"label":label, "value": result}
    return HttpResponse(json.dumps(dic), content_type = "application/json")

def visualize(request):
    if len(request.FILES) == 0:
        return
    file = request.FILES["getCSV"]
    df = pd.read_csv(file)
    df = df.fillna("NaN")
    result = df["value"].values.tolist()
    label = df["time"].values.tolist()
    dic = {"label":label, "value": result}
    return HttpResponse(json.dumps(dic), content_type = "application/json")


def save(request):
    uid = request.POST["uid"]
    naomiIMP = getModel(uid)

    result_df = naomiIMP.df.copy()
    result = naomiIMP.predict_result()
    result = naomiIMP.scaler.inverse_transform(result.reshape(-1,1)).squeeze()
    result_df["value"] = result
    src = "media/result_" + uid + ".csv"
    result_df.to_csv(src)
    dic = {"src" : src}
    return HttpResponse(json.dumps(dic), content_type = "application/json")

def delete(request):
    uid = request.POST["uid"]
    file = "media/result_" + uid + ".csv"
    
    if os.path.isfile(file):
        os.remove(file)
    if uid in modelDic:
        del(modelDic[uid])
    dic = {"ret":True}
    return  HttpResponse(json.dumps(dic), content_type = "application/json")