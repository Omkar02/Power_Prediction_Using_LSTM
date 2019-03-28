from ubidots import ApiClient
######Create an "API" object
api = ApiClient("BBFF-31019c8d2021bf324cbaf0d8db74df7d33f")
######Create a "Variable" object
graph = api.get_variable("5c9847d31d8472615c845a0f")
battery = api.get_variable("5c98479f1d847260650f0285")
# graph_y = api.get_variable("5c9760b11d847225f4b15e7d")
import time
battery.save_value({'value': 100})
graph.remove_values(0,1483228800000)
# print(battery.get_values())

def val_sender_ubi(the_val,result_m):
    remanin = 100
    for index, stuff in enumerate(the_val):
        print(time.time())
        stuff_1 = stuff + result_m
        remaning_batt = (stuff_1/5000)*100
        remanin = remanin - remaning_batt
        graph.save_value({'value': stuff_1})
        battery.save_value({'value': remanin})
        print(index,'has transmited',stuff_1)
        # print(stuff_1,list(range(len(the_val)))[index])
        # graph_y.save_value({'value': list(range(len(the_val)))[index]})
