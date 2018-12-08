
# coding: utf-8

# In[48]:


from pandas.io.json import json_normalize
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import sys
import os.path
import argparse

def read_json_fun(file):
    data = []
    temp = []
    data_str = open(file).read()
    temp = data_str.splitlines()
    num = len(temp)
    for i in range(num):
        str = temp[i]
        data.append(json.loads(str))
    return data



def Generator(opts):
    #df = pd.read_excel('Markov.xlsx')
    col = ['file_name','label','type','sa','da','sp','dp','pr','tls_scs','tls_ext_server_name','tls_c_key_length','http_content_type','http_user_agent','http_accept_language','http_server','http_code','dns_domain_name','dns_ttl','dns_num_ip','dns_domain_rank','formean','forvar','backmean','backvar','duration', 'tot_forpkts', 'tot_backpkts', 'tot_forpktsize', 'tot_backpktsize', 'maxforpktsize', 'minforpktsize', 'maxbackpktsize', 'minbackpktsize', 'numbytepersec', 'foriptmean', 'foriptstd', 'backiptmean', 'backiptstd', 'totfoript', 'totbackipt', 'maxfoript', 'minfoript', 'maxbackipt', 'minbackipt', 'numforpktpersec', 'numbackpktpersec', 'numpktpersec', 'ttlout', 'ttlin']
    for i in range(20):
        for j in range(20):
            splt_str = 'splt_' + str(i) + '_' + str(j) 
            col.append(splt_str)

    for i in range(256):
        dist_str = 'dist_' + str(i)
        col.append(dist_str)
    col.append('entropy')

   # df = pd.DataFrame(columns = col)

    #print(df.columns)
    inputfile=opts.source_data_path
    print('python input:',inputfile)
    data = read_json_fun(inputfile)
    fn=os.path.basename(inputfile)
    title=os.path.splitext(fn)[0]
    print('file = ',title)
    
    if(opts.type == 'default'):
        Type = Catgo(title, opts)
    else:
        Type = opts.type

    print('type = ',Type)
    NumIter = len(data)
    for i in range(0,len(data)):
        df = pd.DataFrame(columns = col)
        Basic_Info(data, i, df, Type, title)
        Marcov(data, i, df)
        TLS(data, i, df)
        http(data, i ,df)
        dns(data, i ,df)
        Byte_dist(data, i, df)
        statisticfeature(data,i ,df)
        iptfeature(data, i, df)
        ttl(data, i, df)
        if(opts.is_malware == 1):
            df.loc[i,'label'] = 1
        else:
            df.loc[i,'label'] = 0
        if(i%100 == 0):
            print('Processing:{}/{}'.format(i,NumIter))

    #Saving Dataframe into csv
        out_file = opts.output_path
        if(os.path.isfile(out_file)):
            df.to_csv(out_file, mode = 'a',header = False)
        else:
            df.to_csv(out_file, header = col)

def ttl(data, i, df):
    ttlout = 0
    ttlin = 0
    if (data[i].__contains__('ip')):
        if (data[i]['ip'].__contains__('out')):
            ttlout = data[i]['ip']['out']['ttl']
        if (data[i]['ip'].__contains__('in')):
            ttlin = data[i]['ip']['in']['ttl']
    df.loc[i,'ttlout'] = ttlout
    df.loc[i,'ttlin'] = ttlin
    #return ttlout, ttlin
            
def statisticfeature(data, i, df):
    formean = 0
    backmean = 0
    forvar = 0
    backvar = 0
    forbyte = []
    backbyte = []
    forsum = 0
    backsum = 0
    firstdir = "0"
    duration = 0
    tot_forpkts = 0
    tot_backpkts = 0
    tot_forpktsize = 0
    tot_backpktsize = 0
    maxforpktsize = 0
    minforpktsize = 0
    maxbackpktsize = 0
    minbackpktsize = 0
    numbytepersec = 0
    if(data[i].__contains__("packets")):
        packets = data[i]['packets']
        j = 0
        if(len(packets) != 0):
            while j < len(packets):
                if(packets[j].__contains__('dir') and packets[j].__contains__('b')):
                    firstdir = packets[j]['dir']
                    break;
                else:
                    j = j + 1
            k = 0
            for k in range(len(packets)):
                if(packets[k].__contains__('dir') and packets[k].__contains__('b')):
                    if(packets[k]['dir'] == firstdir):
                        forbyte.append(packets[k]['b'])
                        forsum = forsum + packets[k]['b']
                    else:
                        backbyte.append(packets[k]['b'])
                        backsum = backsum + packets[k]['b']
                    if(packets[k].__contains__('ipt')):
                        duration = duration + packets[k]['ipt']
            if(len(forbyte) == 0):
                formean = 0
            else:
                formean = forsum/len(forbyte)
            if(len(backbyte) == 0):
                backmean = 0
            else:
                backmean = backsum/len(backbyte)
            if(formean != 0):
                tmpsum = 0
                for m in range(len(forbyte)):
                    tot_forpktsize = tot_forpktsize + forbyte[m]
                    tmpsum = tmpsum + (forbyte[m] - formean) * (forbyte[m] - formean)
                forvar = (tmpsum/len(forbyte)) ** 0.5
                tot_forpkts = len(forbyte)
                maxforpktsize = max(forbyte)
                minforpktsize = min(forbyte)
            if(backmean != 0):
                tmpsum = 0
                for m in range(len(backbyte)):
                    tmpsum = tmpsum + (backbyte[m] - backmean) * (backbyte[m] - backmean)
                    tot_backpktsize = tot_backpktsize + backbyte[m]
                backvar = (tmpsum/len(backbyte)) ** 0.5
                tot_backpkts = len(backbyte)
                maxbackpktsize = max(backbyte)
                minbackpktsize = min(backbyte)
            if duration != 0:
                numbytepersec = (tot_backpktsize + tot_forpktsize) / duration
    df.loc[i,'formean'] = formean
    df.loc[i,'forvar'] = forvar
    df.loc[i,'backmean'] = backmean
    df.loc[i,'backvar'] = backvar
    df.loc[i,'duration'] = duration
    df.loc[i,'tot_forpkts'] = tot_forpkts
    df.loc[i,'tot_backpkts'] = tot_backpkts
    df.loc[i,'tot_forpktsize'] = tot_forpktsize
    df.loc[i,'tot_backpktsize'] = tot_backpktsize
    df.loc[i,'maxforpktsize'] = maxforpktsize
    df.loc[i,'minforpktsize'] = minforpktsize
    df.loc[i,'maxbackpktsize'] = maxbackpktsize
    df.loc[i,'minbackpktsize'] = minbackpktsize
    df.loc[i,'numbytepersec'] = numbytepersec

   # return formean, forvar, backmean, backvar, duration
    #return tot_forpkts, tot_backpkts, tot_forpktsize, tot_backpktsize, maxforpktsize, minforpktsize, maxbackpktsize, minbackpktsize, numbytepersec

def iptfeature(data, i, df):
    foriptmean = 0
    backiptmean = 0
    foriptstd = 0
    backiptstd = 0
    foript = []
    backipt = []
    foriptsum = 0
    backiptsum = 0
    totfoript = 0
    totbackipt = 0
    maxfoript = 0
    minfoript = 0
    maxbackipt = 0
    minbackipt = 0
    firstdir = "0"
    numforpktpersec = 0
    numbackpktpersec = 0
    numpktpersec = 0
    if(data[i].__contains__("packets")):
        packets = data[i]['packets']
        j = 0
        if(len(packets) != 0):
            while j < len(packets):
                if(packets[j].__contains__('dir') and packets[j].__contains__('ipt')):
                    firstdir = packets[j]['dir']
                    break;
                else:
                    j = j + 1
            k = 0
            for k in range(len(packets)):
                if(packets[k].__contains__('dir') and packets[k].__contains__('ipt')):
                    if(packets[k]['dir'] == firstdir):
                        foript.append(packets[k]['ipt'])
                        foriptsum = foriptsum + packets[k]['ipt']
                    else:
                        backipt.append(packets[k]['ipt'])
                        backiptsum = backiptsum + packets[k]['ipt']
            if(len(foript) == 0):
                foriptmean = 0
            else:
                foriptmean = foriptsum/len(foript)
            if(len(backipt) == 0):
                backiptmean = 0
            else:
                backiptmean = backiptsum/len(backipt)
            if(foriptmean != 0):
                tmpsum = 0
                for m in range(len(foript)):
                    tmpsum = tmpsum + (foript[m] - foriptmean) * (foript[m] - foriptmean)
                    totfoript = totfoript + foript[m]
                foriptstd = (tmpsum/len(foript)) ** 0.5
                maxfoript = max(foript)
                minfoript = min(foript)
                if totfoript != 0:
                    numforpktpersec = len(foript)/totfoript
            if(backiptmean != 0):
                tmpsum = 0
                for m in range(len(backipt)):
                    tmpsum = tmpsum + (backipt[m] - backiptmean) * (backipt[m] - backiptmean)
                    totbackipt = totbackipt + backipt[m]
                backiptstd = (tmpsum/len(backipt)) ** 0.5
                maxbackipt = max(backipt)
                minbackipt = min(backipt)
                if totbackipt != 0:
                    numbackpktpersec = len(backipt)/totbackipt
            if totbackipt != 0 or totfoript != 0:
                numpktpersec = (len(foript) + len(backipt)) / (totfoript + totbackipt)
    
    df.loc[i,'foriptmean'] = foriptmean
    df.loc[i,'totbackipt'] = totbackipt
    df.loc[i,'maxfoript'] = maxfoript
    df.loc[i,'minfoript'] = minfoript
    df.loc[i,'maxbackipt'] = maxbackipt
    df.loc[i,'minbackipt'] = minbackipt
    df.loc[i,'numforpktpersec'] = numforpktpersec
    df.loc[i,'numbackpktpersec'] = numbackpktpersec
    df.loc[i,'numpktpersec'] = numpktpersec
    df.loc[i,'totfoript'] = totfoript
    df.loc[i,'foriptstd'] = foriptstd
    df.loc[i,'backiptmean'] = backiptmean
    df.loc[i,'backiptstd'] = backiptstd
    
    #return foriptmean, foriptstd, backiptmean, backiptstd
    #return totfoript, totbackipt, maxfoript, minfoript, maxbackipt, minbackipt, numforpktpersec, numbackpktpersec, numpktpersec

def Byte_dist(data, i, df):
    bdlist = []
    su = 0
    if data[i].__contains__('byte_dist'):
        pakpd=data[i]['byte_dist']
        for n in pakpd:
            bdlist.append(n)
            su += n
        if su <= 0:
            bdlist = np.zeros(256)
        else:
            bdlist = np.array(bdlist) / su
    else:
        bdlist = np.zeros(256)

    for q in range(256):
        dist_str = 'dist_' + str(q)
        df.loc[i,dist_str] = bdlist[q]
    if data[i].__contains__('entropy'):
        df.loc[i,'entropy'] = data[i]['entropy']
    else:
        df.loc[i,'entropy'] = -1
def http(data, i, df):
    http_col = ['http_content_type','http_user_agent','http_accept_language','http_server','http_code']
    if data.__contains__("http"):
        http = data['http'][0]
        if http.__contains__('in'):
            code = False
            server = False
            content = False
            for element in http['in']:
                if element.__contains__("code"):
                    df.loc[i,'http_code'] = element['code']
                    code = True
                if element.__contains__("Server"):
                    df.loc[i,'http_server'] = element["Server"]
                    server = True
                if element.__contains__("Content-Type"):
                    df.loc[i,'http_content_type'] = element['Content-Type']
                    content = True
            if not code:
                df.loc[i,'http_code'] = 'http_code_NULL'
            if not server:
                df.loc[i,'http_server'] = 'http_server_NULL'
            if not content:
                df.loc[i,'http_content_type'] = 'http_content_type_NULL'

        else:
            in_col = ['http_content_type','http_server','http_code']
            df.loc[i,in_col] = [k+'NULL' for k in in_col]
        if http.__contains__('out'):
            agent = False
            lang = False
            for element in http['out']:
                if element.__contains__("User-Agent"):
                    df.loc[i,'http_user_agent'] = element["User-Agent"]
                    agent = True
                if element.__contains__("Accept-Language"):
                    df.loc[i,'http_accept_language'] = element["Accept-Language"]
                    lang = True
            if not agent:
                df.loc[i,'http_user_agent'] = 'http_user_agent_NULL'
            if not lang:
                df.loc[i,'http_accept_language'] = 'http_accept_language_NULL'
        else:
            out_col = ['http_user_agent','http_accept_language']
            df.loc[i,out_col] = [k+'NULL' for k in out_col]
    else:
        df.loc[i,http_col] = [k+'NULL' for k in http_col]
    return 

def dns(data, i, df):
    dns_col = ['dns_domain_name','dns_ttl','dns_num_ip','dns_domain_rank']
    if data.__contains__("linked_dns"):
        dns = data['linked_dns']
        num_ip = 0

        if dns.__contains__('dns'):
            if dns['dns'][0].__contains__('rn'):
                df.loc[i,'dns_domain_name'] = dns['dns'][0]['rn']
            else:
                df.loc[i,'dns_domain_name'] = 'dns_domain_name_NULL'
            if dns['dns'][0].__contains__('rr'):
                lis = dns['dns'][0]['rr']
                ttl = -1
                for element in lis:
                    if ttl == -1 and element.__contains__('cname'):
                        ttl = element['ttl']
                    if element.__contains__('a'):
                        num_ip+=1
                if ttl == -1:
                    df.loc[i,'dns_ttl'] = 'dns_ttl_NULL'
                else:
                    df.loc[i,'dns_ttl'] = ttl
                    
                df.loc[i,'dns_num_ip'] = num_ip
                df.loc[i,'dns_domain_rank'] = 'dns_domain_rank_NULL'
            else:
                dns_col = ['dns_ttl','dns_num_ip','dns_domain_rank']
                df.loc[i,dns_col] = [k+'NULL' for k in dns_col]
        else:
            #df.loc[i,dns_col] = 'NULL'
            df.loc[i,dns_col] = [k+'NULL' for k in dns_col]
    else:
        #df.loc[i,dns_col] = 'NULL'
        df.loc[i,dns_col] = [k+'NULL' for k in dns_col]
    return 
def TLS(data, i, df):

    scs_str = 'scs_str_NULL'
    server_name_str = 'server_name_str_NULL'
    c_key_len = '0'

    if data[i].__contains__('tls'):

        tls = data[i]['tls']
        if tls.__contains__('scs'):
            scs_str = tls['scs']
        if tls.__contains__('c_key_length'):
            c_key_len = tls['c_key_length'] 
        if tls.__contains__('c_extensions'):
            tls_ext = tls['c_extensions']
            if(tls_ext[0].__contains__('server_name')):
                server_name_str = tls_ext[0]['server_name']

    df.loc[i,'tls_scs'] = scs_str
    df.loc[i,'tls_ext_server_name'] = server_name_str
    df.loc[i,'tls_c_key_length'] = c_key_len

def Basic_Info(data, i, df, Type, file_name):

    if data[i].__contains__('sa'):
        df.loc[i,'sa'] = data[i]['sa']
    else:    
        df.loc[i,'sa'] = 'NULL'

    if data[i].__contains__('da'):
        df.loc[i,'da'] = data[i]['da']
    else:    
        df.loc[i,'da'] = 'NULL'
    
    if data[i].__contains__('sp'):
        df.loc[i,'sp'] = data[i]['sp']
    else:    
        df.loc[i,'sp'] = 'NULL'
    
    if data[i].__contains__('dp'):
        df.loc[i,'dp'] = data[i]['dp']
    else:    
        df.loc[i,'dp'] = 'NULL'
    
    if data[i].__contains__('pr'):
        df.loc[i,'pr'] = data[i]['pr']
    else:    
        df.loc[i,'pr'] = 'NULL'
    
    df.loc[i,'type'] = Type
    df.loc[i,'file_name'] = file_name

def Catgo(t, opts):
    if(opts.is_malware==1):
        return 'malware'
    t = t.lower()
    print('t=',t)
    if (t.find('email') !=  -1):
        return 'email'
    elif(t.find('chat') !=  -1):
        return 'chat'
    elif(t.find('youtube')!=-1 or t.find('vimeo')!=-1 or t.find('video')!=-1 or t.find('netflix')!=-1 or t.find('spotify')!=-1 ):
        return 'stream'
    elif(t.find('ftp')!=-1 or t.find('scp')!=-1 or t.find('file')!=-1):
        return 'file_trans'
    elif(t.find('audio')!=-1 or t.find('voip')!=-1):
        return 'voip'
    elif(t.find('torrent')!=-1):
        return 'trap2p'
    else:
        print('browsing?')
        return 'browsing'


def Marcov(data, i, df):

    d=json.dumps(data[i], indent=4)
    #print(d)
    if data[i].__contains__('packets'):
        #print(1)
        packets = data[i]['packets']
        clas = []
        count = [[0]*20 for i in range(20)]
        matrix = [[0]*20 for i in range(20)]
        #print(len(packets))
        if len(packets) > 1:
            #print(len(packets))
            for j in range(len(packets)):
                if packets[j].__contains__('b'):
                    byte = packets[j]['b']
                    if packets[j]['dir'] == '>':
                        if byte == 1500:
                            clas.append(9)
                        elif byte < 1500:
                            clas.append(int(byte/150))
                    else:
                        if byte == 1500:
                            clas.append(19)
                        elif byte < 1500:
                            clas.append(int(byte/150)+10)
            for j in range(len(clas)-1):
                count[clas[j+1]][clas[j]] += 1
            total = 0
            for j in range(20):
                for k in range(20):
                    total = total + count[j][k]
            for j in range(20):
                for k in range(20):
                    if total > 0:
                        matrix[j][k] = count[j][k]/total
    else:
        matrix = np.zeros([20,20])
            
            
            
    for j in range(20):
        for k in range(20):
            splt_str = 'splt_' + str(j) + '_' + str(k) 
            df.loc[i,[splt_str]] = matrix[j][k]
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Making Traffic Table')
    parser.add_argument('--source_data_path', type=str,
                        default='../LinksToPCAP/youtube1.json', dest='source_data_path',
                        help='Path to source data')
    parser.add_argument('--output_path', type=str,
                        default='table.csv', dest='output_path',
                        help='Path to output')
    parser.add_argument('--is_malware', type=int,
                        default=0, dest='is_malware',
                        help='Label it if it is malware')
    parser.add_argument('--type', type=str,
                        default="default", dest='type',
                        help='The type of file')
    opts = parser.parse_args()

    #inputfile=opts.source_data_path
    #print('python input:',inputfile)
    Generator(opts)

