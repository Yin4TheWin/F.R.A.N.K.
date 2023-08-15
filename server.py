
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, jsonify
from llama_index import VectorStoreIndex, Prompt, Document, ServiceContext, set_global_service_context
from flask_cors import CORS, cross_origin 
from llama_index.llms import OpenAI
from datetime import date

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS']='Content-Type'

llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)

@app.route('/', methods=['GET'])
def hello():
    print('get')
    return 'Hello, World!'

@app.route('/query', methods=['POST'])
def query_model():
    data = request.get_json()
    job_lists = ["Here's the jobs I've applied to, but not heard back from, in JSON format:\n"+str(data['applied']), "Here's the jobs I'm interviewing for in JSON format:\n"+str(data['interviewing']), "Here's the jobs I've been offered in JSON format:\n"+str(data['offered']), "Here's the jobs I've been rejected from in JSON format:\n"+str(data['rejected'])]
    documents = [Document(text=t) for t in job_lists]
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    TEMPLATE_STR = (
    "You are a friendly job search assistant named F.R.A.N.K., which stands for Friendly Robotic Assistant for Navigating_career Knowledge. When you introduce yourself, make a joke about how Navigating_career is one word.\n"
    "Given a list of jobs I've applied to, your purpose is to answer any questions I have about these jobs, or about the job searching process. If something unrelated to these is asked, say you cannot answer.\n"
    "If I ask you to generate content for a specific job, perform a fuzzy search on all my job lists for information regarding that job.\n"
    "If I ask for information about deadlines, look for a \"deadlines\" field for all jobs, and return the relevant deadlines based on today's date (which is "+str(date.today())+") and the timeframe of my request. If I ask for upcoming or past deadlines, apologize and say you are still learning how to compare dates, so you cannot answer.\n"
    "If you are ever about to say this exact text: [Recruiter Name], and the job you are talking about has a recruiter, replace [Recruiter Name] with the recruiter's name. For example, if the recruiter's name is John Smith, say 'John Smith' instead of '[Recruiter Name]'.\n"
    "Here are my relevant job applications: \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n")

    QA_TEMPLATE = Prompt(TEMPLATE_STR)
    
    # index = GPTListIndex.from_documents(documents)
    query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)

    response = query_engine.query(data['query'])
    response_str = "It looks like your list is empty! Please add some jobs to your list before asking me a question. If you believe this is a mistake, contact my maker at franklinyin.nj@gmail.com." if str(response)=="None" else str(response).replace('[Your Name]', data['name'])
    return jsonify({'response': response_str})


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=3000, url_scheme='https')