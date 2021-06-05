from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import speech_recognition as sr
import pyttsx3
from collections import Counter
import re
import math
import pyaudio as py
from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification
from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification

app = Flask(__name__)
bot = ChatBot(name = 'InterviewBot',
                  read_only = True,
                  logic_adapters = ["chatterbot.logic.BestMatch"],                 
                  storage_adapter = "chatterbot.storage.SQLStorageAdapter")

third_ans="Security groups are tied to an instance whereas Network ACLs are tied to the subnet.Security groups are stateful.Network ACLs are stateless.Security group support allow rules only.Network ACL support allow and deny rules.Subnet can have only one NACL, whereas Instance can have multiple Security groups.Security group first layer of defense, whereas Network ACL is second layer of the defense.All rules in a security group are applied whereas rules are applied in their order in Network ACL."
fourth_ans="You cannot create a VPC peering connection between VPCs that have matching or overlapping IPv4 or IPv6 CIDR blocks.You have a quota on the number of active and pending VPC peering connections that you can have per VPC.You have a quota on the number of active and pending VPC peering connections that you can have per VPC.You cannot have more than one VPC peering connection between the same two VPCs at the same time."
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz", "textual_entailment")
correctCount=0
scoreDetails=" "
greet_conversation = [
    "Hello",
    "Hi there!",
    "How are you doing?",
    "I'm doing great.",
    "That is good to hear",
    "Thank you.",
    "You're welcome."
    "bye",
    "Goodbye!",
    "what do you do?",
    "I'm a Interview Bot"

]
 
first_ques = [
    "ask me",
    "Explain the Relation between Region and Availability Zones",
    "Start interview",
    "Explain the Relation between Region and Availability Zones",
    "start",
    "Explain the Relation between Region and Availability Zones"
]
 
second_ques = [
    "Each Region is a separate geographic area. Each Region has multiple, isolated locations known as Availability Zones",
    "Explain all the differences between AWS Security Group vs NACL",
    "Availability Zones in a Region are connected through low-latency links",
    "Explain all the differences between AWS Security Group vs NACL"
]
third_ques=[third_ans,
    "What are the constraints to do VPC Peering"
   ]

fourth_ques=[fourth_ans,
             "Explain CI/CD services in AWS?"]

fifth_ques=["A CI/CD pipeline helps you automate steps in your software delivery process, such as initiating code builds, running automated tests, and deploying to a staging or production environment",
             "Deployment types available in Code Deploy?"]

sixth_ques=["in-place deployments and blue/green deployments",
            "What is a Commit in Git?"]

network1=["The commit command is used to save your changes to the local repository.",
            "Explain OSI layers"]
network2=["The Open Systems Interconnection model is a conceptual model that characterises and standardises the communication functions of a telecommunication or computing system without regard to its underlying internal structure and technology.",
           "Ports & protocols: HTTP, DNS, FTP, SMTP, MySQL, RDP" ]

network3=["http:80,dns:53,ftp:21,smtp:25,465,587,mysql:3306,rdp:3389",
          "Calculate No. of IPs for a given CIDR"]
ninth_ques=["classful networking. Subtract the number of network bits from 32. Raise 2 to that power and subtract 2 for the network and broadcast addresses.",
            "Difference between Task Definition and Service in ECS"]

tenth_ques=["Task Definition is a blueprint that describes how a docker container should launch.Service defines long running tasks of the same Task Definition.",
            "What is the relation between a task and a container?"]

docker1=["A Container Instance can run many tasks from the same or different Services.",
         "How is Containerization different from Virtualization?"]


docker2=["Virtualization enables you to run multiple operating systems on the hardware of a single physical server, while containerization enables you to deploy multiple applications using the same operating system on a single virtual machine or server.",
         "What happens when you do Docker build? Explain all the steps in detail without skipping anything."]
docker3=["The docker build command builds Docker images from a Dockerfile and a context. A build’s context is the set of files located in the specified PATH or URL. The build process can refer to any of the files in the context.",
         " What is Multistage build and why should we do it?"]
terraform1=["Multi-stage builds are a method of organizing a Dockerfile to minimize the size of the final container, improve run time performance, allow for better organization of Docker commands and files, and provide a standardized method of running build actions.",
            "Can terraform replace SCM tools like Chef, Puppet and Ansible?"]

terraform2=["Terraform is a better choice than a configuration management tool.Yes",
            "Which syntax is used to write terraform files?"]

terraform3=["HashiCorp Configuration Language,HCL.",
            "What is the difference between a provider and a provisioner in terraform?"]

jenkins1=["A provider is responsible for understanding API interactions and exposing resources.Provisioners are used to execute scripts on a local or remote machine as part of resource creation or destruction.",
          "Explain Pipeline in Jenkinsfile"]
jenkins2=["Jenkins Pipeline is a suite of plugins which supports implementing and integrating continuous delivery pipelines into Jenkins.",
          "How a task is related with a playbook?"]
ansible1=["Each playbook is composed of one or more ‘plays’ in a list.The goal of a play is to map a group of hosts to some well defined roles, represented by things ansible calls tasks. At a basic level, a task is nothing more than a call to an ansible module.",
          "What is Ansible Galaxy?"]

ansible2=["Ansible Galaxy is a repository for Ansible Roles that are available to drop directly into your Playbooks to streamline your automation projects",
        "What is a handler in Ansible"]

ansible3=["Handlers are just like regular tasks in an Ansible playbook but are only run if the Task contains a notify directive and also indicates that it changed something",
         "Command in Linux to find the listening TCP ports" ]

linux1=["Netstat,ss,Nmap,Isof",
        "How do you enable and manage firewall in Ubuntu"]
linux2=["execute $ sudo ufw enable",
        "What is the command to list down the network interfaces"]
linux3=["netstat",
        "Congratulations!Your done with the interview",
        "End interview",
        "You have ended the interview",
        "I don't know",
        "If you do not know the answer,please enter a keyword in the question to proceed"]





trainer = ListTrainer(bot)

trainer.train(greet_conversation)
trainer.train(first_ques)
trainer.train(second_ques)
trainer.train(third_ques)
trainer.train(fourth_ques)
trainer.train(fifth_ques)
trainer.train(sixth_ques)
trainer.train(network1)
trainer.train(network2)
trainer.train(network3)
trainer.train(ninth_ques)
trainer.train(tenth_ques)
trainer.train(docker1)
trainer.train(docker2)
trainer.train(docker3)
trainer.train(terraform1)
trainer.train(terraform2)
trainer.train(terraform3)
trainer.train(jenkins1)
trainer.train(jenkins2)
trainer.train(ansible1)
trainer.train(ansible2)
trainer.train(ansible3)
trainer.train(linux1)
trainer.train(linux2)
trainer.train(linux3)

"""def text_to_vector(text):
    WORD = re.compile(r"\w+")
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
def solution(ans,correctAns):
    vector1 = text_to_vector(ans)
    vector2 = text_to_vector(correctAns)
    cosine = get_cosine(vector1, vector2)
    return cosine"""
@app.route("/")
def index():    
    return render_template("indexBot.html")
@app.route("/get")
def get_bot_response():
    a = request.args.get('msg')
    userText = str(bot.get_response(a))
    x = pyttsx3.init()
    li = []
    if len(userText) > 100:
        if userText.find('--') == -1:
            b = userText.split('--')
            # print(b)

    x.setProperty('rate', 120)
    x.setProperty('volume', 100)
    x.say(userText)
    x.runAndWait()
    return userText

@app.route("/record")
def record():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Talk")
        audio_text = r.listen(source)
        print("Time over, thanks")
        try:
            a = r.recognize_google(audio_text)
            print("Text: " + a)
            return a
        except:
            print("Sorry, I did not get that")
            return "Please repeat"

@app.route("/score")
def score():
    global correctCount
    return str(correctCount)+ " out of 25. " + "\n \n \n" +scoreDetails

@app.route("/calScore")
def calculateScore():
    a = request.args.get('msg')
    r = request.args.get('msg2')

    enteredAns=str(a)
    response=str(r)
    coorectAns = ""
    if response==first_ques[1]:
        coorectAns=second_ques[0]
    if response == second_ques[1]:
        coorectAns = third_ques[0]
    if response==third_ques[1]:
        coorectAns=fourth_ques[0]
    if response == fourth_ques[1]:
        coorectAns = fifth_ques[0]
    if response==fifth_ques[1]:
        coorectAns=sixth_ques[0]
    if response == sixth_ques[1]:
        coorectAns = network1[0]
    if response==network1[1]:
        coorectAns=network2[0]
    if response == network2[1]:
        coorectAns = network3[0]
    if response==network3[1]:
        coorectAns=ninth_ques[0]
    if response == ninth_ques[1]:
        coorectAns = tenth_ques[0]
    if response==tenth_ques[1]:
        coorectAns=docker1[0]
    if response == docker1[1]:
        coorectAns = docker2[0]
    if response == docker2[1]:
        coorectAns=docker3[0]
    if response == docker3[1]:
        coorectAns = terraform1[0]
    if response == terraform1[1]:
        coorectAns = terraform2[0]
    if response==terraform2[1]:
        coorectAns=terraform3[0]
    if response == terraform3[1]:
        coorectAns = jenkins1[0]
    if response==jenkins1[1]:
        coorectAns=jenkins2[0]
    if response == jenkins2[1]:
        coorectAns = ansible1[0]
    if response==ansible1[1]:
        coorectAns=ansible2[0]
    if response == ansible2[1]:
        coorectAns = ansible3[0]
    if response == ansible3[1]:
        coorectAns = linux1[0]
    if response==linux1[1]:
        coorectAns=linux2[0]
    if response == linux2[1]:
        coorectAns = linux3[0]
    predictions=predictor.predict(hypothesis=enteredAns,premise=coorectAns)
    l=predictions['label_probs']
    entailment=l[0]
    contradiction=l[1]
    neutral=l[2]
    print("user: ", a)
    print("chatbot :",response, "ques: ", first_ques[1], response==first_ques[1])
    print("enteredAns: ",enteredAns)
    print("coorectAns: ",coorectAns)
    global correctCount
    global scoreDetails
    print(entailment,contradiction,neutral)
    ans=0
    if(coorectAns!=""):
        if (entailment >= 0.3 and neutral>=1.0e-09 and contradiction>=9e-08 ):
            correctCount = correctCount + 1
            ans=1
    
    print("correctCount: ", correctCount)
    if coorectAns!="":
        scoreDetails=scoreDetails+ response +" : "+ str(ans)+ "\n"
    return str(entailment)

if __name__ == "__main__":
    app.run()
