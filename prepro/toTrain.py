import pandas as pd
import numpy as np
import sys

LABEL2DIG = {'chat':0, 'voip':1, 'trap2p':2, 'stream':3, 'file_trans':4, 'email':5}
DIG2LABEL = {v: k for k, v in LABEL2DIG.items()}
#nclass = 12

if __name__=="__main__":
    
    filepath = sys.argv[1]
    #targetpath = sys.argv[2]

    df = pd.read_csv(filepath)
    print('df shape: ',df.shape)
    print('df shape: ',df.columns)
    
    #create column vpn
    df['vpn'] = df.file_name.str.contains('vpn').astype(int)
    
    df = df[(df.type!='browsing')&(df.file_name!='skype_audio1a_test')]
    # Broadcast and Multicast
    df = df[~(df.da.str.contains('224.0.'))]
    df = df[~(df.da.str.contains('239.255.'))]
    df = df[~(df.da.str.contains('255.255.'))]
    # Zzro packet size flow
    df = df[(df.formean!=0)|(df.backmean!=0)]
    # file transfer should not use udp
    df = df[~((df.type=='file_trans')&(df.pr==17))]
    # clean icmp
    df = df[df.pr!=1]
    # clean NetBIOS
    df = df[(df.sp>139)|(df.sp<137)]
    # one packet flow
    df = df[(df.tot_forpkts + df.tot_backpkts) > 1]

    #get y
    type_ = df.type.values
    type_ = [LABEL2DIG[label] for label in type_]
    
    y_train = df.vpn.values * 6 + type_
    #y_train = type_

    print('y_train:', np.shape(y_train))
    np.save('../data/y_train', y_train)
    
    df = df.drop(df.columns[0], axis=1)
    print('df shape: ',df.shape)
    df_drop = df.drop(['sp','dp','sa','da','type','file_name','label','vpn'], axis = 1)
    print('df shape: ', df_drop.shape)
    print(df_drop.columns)
    table = pd.get_dummies(df_drop)
    

    
    #table = table.drop(['label'], axis = 1)
    X_train = table.values
    print('train:', table.shape)
    np.save('../data/X_train', X_train)
    #table.to_csv('train.csv', index = False)
    
    table[0:1].to_csv('ColSample.csv', index = False)
    table[0:1].to_csv('ColSample_.csv')




