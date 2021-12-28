
def num_clip(l,u,x):
    if(x>=l):
        if(x<=u):
            r = x
        else:
            r = u
    else:
        r = l
    return r

def adjacent_coordinates(x,y,s):
    adj = []
    adj.append([x-s,y-s])
    adj.append([x,y-s])
    adj.append([x+s,y-s])
    adj.append([x-s,y])
    adj.append([x+s,y])
    adj.append([x-s,y+s])
    adj.append([x,y+s])
    adj.append([x+s,y+s])
    return adj
