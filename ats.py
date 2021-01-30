import os
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import docx2txt
import PyPDF2 
from pyresparser import ResumeParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import glob
import textract
from pyresparser import ResumeParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
import numpy
import numpy as np
from bs4 import BeautifulSoup as bs
porter = PorterStemmer()
lancaster=LancasterStemmer()

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx','doc'])
currentDirectory = os.getcwd()
UPLOAD_JOB_DESCRIPTION = currentDirectory+'/job_description'
if not os.path.exists(UPLOAD_JOB_DESCRIPTION):
    os.mkdir(UPLOAD_JOB_DESCRIPTION)
 
UPLOAD_RESUMES = currentDirectory+'/resumes'
if not os.path.exists(UPLOAD_RESUMES):
    os.mkdir(UPLOAD_RESUMES)

app = Flask(__name__)
app.config['UPLOAD_JOB_DESCRIPTION'] = UPLOAD_JOB_DESCRIPTION
app.config['UPLOAD_RESUMES'] = UPLOAD_RESUMES

    #return render_template('multiple_files.html')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    return render_template('multiple_files.html')
    
@app.route('/dropdown', methods=['POST'])
def dropdown():
    skills={}
    ds3=[]
    final3={}
    main3={}
    education=[]
    edu_matching=[]
    exp_final=[]

    edu_abb={'MCA':['MCA','M.C.A','M.C.A.'],'BACHELORS_Tech':['B-TECH', 'B.TECH','BTECH','B.TECH.'],'MASTERS_Tech':['M-TECH', 'M.TECH','MTECH','M.TECH.'],'POST_GRADUATION':['M.E', 'M.E.','M-TECH', 'M.TECH', 'MTECH','MCA','MBA','M.TECH.'],'UNDER_GRADUATION':['B.E.', 'B.E', 'BS', 'B.S','B-TECH','BSC','B.SC','B.SC.', 'B.TECH','BTECH','B.TECH.'],'BACHELORS_Science':['BSC','B.SC','B.SC.'],'GRADUATION':['M.E', 'M.E.','M-TECH', 'M.TECH', 'MTECH','MCA','MBA','M.TECH.','B.E.', 'B.E', 'BS', 'B.S','B-TECH','BSC','B.SC','B.SC.', 'B.TECH','BTECH','B.TECH.'],}

    technologies=['java','python','.net','jquery','c#.net','vb.net','ado.net','mvc','jquery','javascript','python','flask','sql',              'asp.net','c#','azure','aws','agile','react','mysql','reactnative','cordova','ionic','angular',
'kendo','machinelearning','artificialinteligence','ajax','vb.net','ado.net','neuralnetworks',
'webapi','sqlserver','html5','typescript','bootstarp','restapi','api','webtechnologies',
'paymentgateway','sqllite','html','json','googlemaps','http','rest','soap','webapplication',
'automation','deploy','flutter','unity3d','tilemap','physicseditor','visualstudio','andriodstudio',
'multiplayergame','gamedesign','levelediting','designexperience','waterfall','vmodel',
'waterfallmodel','svn','tfs','git','wpf','testcases','testcase','unity','unityframework','gamedeveloper','csharp','csharpe',
'gamedesigner','textures','animation','gui','sdk','backend','server','mobilegame','webgame',
'gamephysics','particlesystems','AR','VR','multiplayer','3dgames','oops','objecdoriented','certification',
'spine2d', 'dragonbones', 'adobephotoshop','adobeanimate', 'tileeditor','soundediting','xcode',
'admob', 'firebase','ironsource','localization', 'playerservices','createlogs','review','analysis','aptitude','communicationskills','criticalthinking','logicalThinking','analytical',
'problemsolving','teamPlayer','selfmotivation','flexible',
'determination','persistence','quicklearner','timemanagement','honesty',
'leadership','decisionMaking','englishWritten',
'attitude','dependability','flexible','takingResponsibility','conflictresolution',]
    exp=['experience','knowledge','idea','language','developed','involved','good','programme','technical','known']
    year=['year','month','years','months']
    words=['machine learning','artificial inteligence','java script','j query','neural networks','angular js','node js','react js','web api','sql server','react native',
'web technologies','rest api','type script','payment gateway','sql lite','google maps',
'web application','unity 3d','physics editor','tile map','visual studio','andriod studio','multiplayer game','game design','level editing','design experience','v model','waterfall model','test cases','test case','unity framework','game developer','game designer','mobile game','web game','game physics','particle systems','spine 2d', 'dragon bones',
 'adobe photoshop','adobe animate', 'tile editor','sound editing','iron source','player services','create logs','communication skills','critical thinking','logical Thinking',
'problem solving','team Player','self motivation','quick learner','time management','decision Making','english Written','taking Responsibility','conflict resolution','c sharp','c sharpe']
    shortcuts={'ai':'artificialinteligence','ml':'machinelearning','nlp':'naturallanguageprocessing','js':'javascript','csharp':'c#','csharpe':'c#'}              
    if request.method == 'POST':
        resume_all_names = request.files.getlist('resume_files[]')
        education = request.form.getlist('edu')
        radio_rating = request.form.get('rating_radio')
        radio_exp = request.form.get('exp_radio')
        experiences = []
        for resume_name in resume_all_names:
            if resume_all_names and allowed_file(resume_name.filename):
                resume_name_filename = secure_filename(resume_name.filename)
                resume_name.save(os.path.join(app.config['UPLOAD_RESUMES'], resume_name_filename))


        for k,v in request.form.items():
            if k.startswith( 'kk-' ):
                ll=k.split('-')
                experiences.append((ll[1],v))
        similyfy_porter={}
        for key, value in request.form.items():
            if key in technologies:
                skills[porter.stem(key)]=value
                similyfy_porter[porter.stem(key)]=key
        ee=[]
        for i in edu_abb.keys():
            if i in education:
                for k in edu_abb[i]:
                    ee.append(k)
        exp1=[]
        year1=[]
        technologies1=[]
        for k in exp:
            exp1.append(porter.stem(k))
        for k in year:
            year1.append(porter.stem(k))
        for k in technologies:
            technologies1.append(porter.stem(k))
        files=[]
        final={}
        final4={}
        currentDirectory = os.getcwd()
        for file in resume_all_names: #glob.glob("/Users/pradeep/Documents/jupyter/profilesforproject/1.docx"):
            result={}
            result3={}
            #data1 = ResumeParser(currentDirectory+"/resumes/"+file.filename).get_extracted_data()
            ll=[]
            lll=[]
            resume_name_filename = secure_filename(file.filename)
            #print('resume_name_filenameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',resume_name_filename)
            try:
              text = textract.process(currentDirectory+"/resumes/"+resume_name_filename)
              #print('normallllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll',text)

            except: 
             try:             
              soup = bs(open(currentDirectory+"/resumes/"+resume_name_filename).read(),'lxml')
              tmpText = soup.get_text()
              text = "".join("".join(tmpText.split('\t')).split('\n')).encode('utf-8').strip()
        
              #print('souppppppppppppppppppppppppppppppppppp',text)
             except:
              sample='sample'
              text=sample.encode('utf-8')
            name=(file.filename.split('/')[-1]).rsplit('.',1)[0]
            text = text.decode("utf-8")

            text=text.replace(',', ' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('+',' ').replace('|',' ').replace('\n',' ')
            my_text=text.lower().split('.\n')
            
            final3[name]=text
            ds3.append(text)
            ll3=[]
            for i in my_text:
             kk=i.split()
             kk1=[]
             for value,term in enumerate(kk):
                  if term.upper() in ee:
                     edu_matching.append((name ,term.upper()))
                  if value+2<len(kk):
                      if kk[value]+' '+kk[value+1] in words:
                        term=kk[value]+''+kk[value+1]
                  if term in shortcuts.keys():
                   kk1.append(porter.stem(shortcuts[term]))
                  else:
                   kk1.append(porter.stem(term))

             ll3.append(kk1)
            for ll2 in ll3:
             for k,j in enumerate(ll2):
                   for l in skills.keys():
                    if j==l:
                        flag1='TRUE'
                        flag2='TRUE'
                        front=ll2[:k+1]
                        rare=ll2[k+1:]
                        for k1,k2 in enumerate(front[::-1]):
                         try:
                            if k2=='year' and flag1=='TRUE':
                              lll.append((j,front[-k1-2]+' '+front[-k1-1]))
                              ll.append((j,int(float(front[-k1-2])*12)))
                              flag1='FLASE'
                            if k2=='month' and flag1=='TRUE':
                                #print(j,front[-k1-2],front[-k1-1])
                                lll.append((j,front[-k1-2]+' '+front[-k1-1]))
                                ll.append((j,int(float(front[-k1-2])*1)))
                                flag1='FLASE'
                                
                            if k2 in exp1 and flag2=='TRUE':
                              #print(j,front[front.index(k)-1],front[front.index(k)])
                              lll.append((j,'knowledge'))
                              #print(j,front[-k1-2])
                              ll.append((j,8))
                              flag2='FLASE'
                         except:
                           pass
                        for r,k4 in enumerate(rare):
                          try:
                            if k4=='year' and flag1=='TRUE':
                              #print(j,rare[r])
                              lll.append((j,rare[r-1]+' '+rare[r]))
                              ll.append((j,int(float(rare[r-1])*12)))
                              flag1='FLASE'
                            if k4=='month' and flag1=='TRUE':
                              #print(j,rare[r])
                              ll.append((j,rare[r-1]+' '+rare[r]))
                              ll.append((j,int(float(rare[r-1])*1)))
                              flag1='FLASE'
                            if k4 in exp1 and flag2=='TRUE':
                              #print(j,front[front.index(k)-1],front[front.index(k)])
                              #print(j,rare[r-1])
                              lll.append((j,'knowledge'))
                              ll.append((j,8))
                              flag2='FLASE'
                          except:
                             pass    
                              
            for key,value in lll:
                if key not in result3.keys():
                    result3[key]=[value]
                else:
                    rr=result3[key]
                    rr.append(value)
                    result3[key]=list(set(rr))

            final4[name]=result3
            df=pd.DataFrame.from_dict(final4)
            final5 = df.replace(np.nan, 'Not Trained', regex=True).to_dict()
            result7=next(iter(final5.values()))
            result8=[]
            for i,v in result7.items():
                result8.append(i)
            for key,value in ll:
              try:
                if key not in result.keys():
                    result[key]=value
                else:
                    rr=result[key]
                    if rr > value:
                        pass
                    else:
                        result[key]=value
              except:
                 pass
            #print(result)
            #result['name']=data1['name']
            #result['email']=data1['email']
            #result['mobile_number']=data1['mobile_number']
            final[name]=result
           
    final1=pd.DataFrame.from_dict(final)
    if radio_exp == 'and':
        final1 = final1.T
        final1 = final1.dropna()
        final1 = final1.T

    final2=final1.replace(np.nan, 'Not Trained', regex=True).to_dict()
    result2=next(iter(final2.values()))
    result6=[]
    for i,v in result2.items():
         result6.append(similyfy_porter[i])
    hello6={}
    for i,v in final.items():
        hello2={}
        for j,k in v.items():
           hello2[j]=1
        hello6[i]=hello2
    df=pd.DataFrame.from_dict(hello6)
    if radio_rating == 'and':
        df = df.T
        df = df.dropna()
        df = df.T
    df=df.replace(numpy.nan, 0 , regex=True)
    df = df.to_dict()
    hello3={}
    rating_result={}
    for i,v in df.items():
        ll=0
        for j,k in v.items() :
            if j in skills.keys():
             ll=ll+(k*int(skills[j]))
        hello3[i]=ll
    rating_result=sorted(hello3.items(), key=lambda x: x[1], reverse=True)
            
    # sorting dataframe based on experience nearest values
    sort_exp=pd.DataFrame.from_dict(final,orient='index')
    ### if radio_exp == and means all technologies must
    if radio_exp == 'and':
        sort_exp = sort_exp.dropna()

    for (k,v) in experiences:
      try:
        if k in sort_exp.columns.values:
            
            if v=='knowledge':
                sort_exp[k]=(sort_exp[k]-8).abs()
                
            else:
                sort_exp[k]=(sort_exp[k]-(float(v)*12)).abs()
      except:
       pass 
    sort_exp=sort_exp.fillna(sort_exp.max()+10, downcast='infer')
    sort_exp["sum"] = sort_exp.sum(axis=1)
    sort_exp=sort_exp.sort_values('sum')
    exp_final=sort_exp.index.values
    return render_template('layer3.html',rating_result = rating_result,edu_matching=edu_matching,experiences=experiences,exp_final=exp_final,result=result6,final=final2)

    
    
    
    
    
@app.route('/multiple_file_upload', methods=['POST'])
def upload_file1():
    job_description_filename=""
    if request.method == 'POST':
        job_description_files = request.files.getlist('job_description_file')
        for job_description_file in job_description_files:
            if job_description_files and allowed_file(job_description_file.filename):
                job_description_filename = secure_filename(job_description_file.filename)
                job_description_file.save(os.path.join(app.config['UPLOAD_JOB_DESCRIPTION'], job_description_filename))
    EDUCATION = [
         'B.E.', 'B.E', 'BS', 'B.S',
         'M.E', 'M.E.', 'MS', 'M.S','BA','B.A','B.A.','BSC',
        'B-TECH', 'B.TECH','BTECH','M-TECH', 'M.TECH', 'MTECH','B.SC','B.SC.',
        'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII','MCA','MBA','BACHELORS','MASTERS','POST_GRADUATION','UNDER_GRADUATION','B.TECH.','M.TECH.','GRADUATION'
    ]
    graduation=['POST_GRADUATION','UNDER_GRADUATION','GRADUATION']
    
    edu_abb={'MCA':['MCA','M.C.A','M.C.A.','MCA.'],'BACHELORS_Tech':['B-TECH','B. TECH' 'B.TECH','BTECH','B.TECH.'],'MASTERS_Tech':['M-TECH','M. TECH', 'M.TECH','MTECH','M.TECH.'],'BACHELORS_Science':['BSC','B.SC','B.SC.''B. SC'] }
    technologies=['java','python','.net','jquery','c#.net','vb.net','ado.net','mvc','jquery','javascript','python','flask','sql',              'asp.net','c#','azure','aws','agile','react','mysql','reactnative','cordova','ionic','angular',
'kendo','machinelearning','artificialinteligence','ajax','vb.net','ado.net','neuralnetworks',
'webapi','sqlserver','html5','typescript','bootstarp','restapi','api','webtechnologies',
'paymentgateway','sqllite','html','json','googlemaps','http','rest','soap','webapplication',
'automation','deploy','flutter','unity3d','tilemap','physicseditor','visualstudio','andriodstudio',
'multiplayergame','gamedesign','levelediting','designexperience','waterfall','vmodel',
'waterfallmodel','svn','tfs','git','wpf','testcases','testcase','unity','unityframework','gamedeveloper','csharp','csharpe'
'gamedesigner','textures','animation','gui','sdk','backend','server','mobilegame','webgame',
'gamephysics','particlesystems','AR','VR','multiplayer','3dgames','oops','objecdoriented','certification',
'spine2d', 'dragonbones', 'adobephotoshop','adobeanimate', 'tileeditor','soundediting','xcode',
'admob', 'firebase','ironsource','localization', 'playerservices','createlogs','review','analysis','aptitude','communicationskills','criticalthinking','logicalThinking','analytical',
'problemsolving','teamPlayer','selfmotivation','flexible',
'determination','persistence','quicklearner','timemanagement','honesty',
'leadership','decisionMaking','englishWritten',
'attitude','dependability','flexible','takingResponsibility','conflictresolution',]
    exp=['experience','knowledge','idea','language','develped','involved','good','programme','technical','known']
    year=['year','month','years','months']
    words=['machine learning','artificial inteligence','java script','j query','neural networks','angular js','node js','react js','web api','sql server','react native',
'web technologies','rest api','type script','payment gateway','sql lite','google maps',
'web application','unity 3d','physics editor','tile map','visual studio','andriod studio','multiplayer game','game design','level editing','design experience','v model','waterfall model','test cases','test case','unity framework','game developer','game designer','mobile game','web game','game physics','particle systems','spine 2d', 'dragon bones',
 'adobe photoshop','adobe animate', 'tile editor','sound editing','iron source','player services','create logs','communication skills','critical thinking','logical Thinking',
'problem solving','team Player','self motivation','quick learner','time management','decision Making','english Written','taking Responsibility','conflict resolution','c sharp','c sharpe']
    shortcuts={'ai':'artificialinteligence','ml':'machinelearning','nlp':'naturallanguageprocessing','js':'javascript','csharp':'c#','csharpe':'c#'}
    exp1=[]
    year1=[]
    technologies1=[]
    for k in exp:
        exp1.append(porter.stem(k))
    for k in year:
        year1.append(porter.stem(k))
    for k in technologies:
        technologies1.append(porter.stem(k))

    currentDirectory = os.getcwd()
    for file in glob.glob(currentDirectory+"/job_description/"+job_description_filename):
        text = textract.process(file)
        text = text.decode("utf-8")

        text=text.replace(',', ' ').replace('\n', ' ').replace('\xa0', ' ').replace('(',' ').replace(')',' ').replace(':',' ').replace('+',' ').replace('|',' ').replace('/',' ').replace('\n',' ')
        my_text=text.lower().split(' ')
        kk1=[]
        for value,term in enumerate(my_text):
              kk=my_text
              if value+2<len(kk):
                  if kk[value]+' '+kk[value+1] in words:
                    term=kk[value]+''+kk[value+1]
              if term in shortcuts.keys():
               kk1.append(shortcuts[term])
              else:
               kk1.append(term)
        my_text=kk1

        hi=Counter(my_text)
    edu=[]
    for value in my_text:
        for k,v in edu_abb.items():
            for term in v:
                if value.upper()==term:
                    edu.append(k)
                    
    ll={}
    list1=[]
    for i,v in enumerate(my_text):
          if my_text[i]=="developer" and my_text[i-1] in technologies:
              list1.append(my_text[i-1])
    list1=list(set(list1))
    for k,v in hi.items():
        if k in technologies:
            ll[k]=v
    ### all technical terms got from job description file have some knowledge 
    exp={}
    for k,v in ll.items():
      exp[k]='knowledge'
      
    final5={}
    low=1
    min=[2,3]
    high=[4,5,6]
    hello=[1,2,3,4,5,6]
    ### rating the  technologies based on count of term(technologiy name)
    for k,v in ll.items():
        if k in list1:
            final5[k]=5
        if v in high:
            final5[k]=4
        if v in min:
            final5[k]=2
        if v==1:
            final5[k]=1
        if v not in hello:
            final5[k]=5
        if k in list1:
            final5[k]=5
    list1=[1,2,3,4,5,6,7,8,9]
    list2=['knowledge',1,2,3,4,5,6,7,8,9]
    edu1=[]
    for i in edu_abb.keys():
        edu1.append(i)
    for i in graduation:
        edu1.append(i)

    return render_template('dropdown.html',edu_abb=edu1,edu=edu,final = final5,list1=list1,list2=list2,exp=exp)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8082)

