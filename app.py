import nltk
import re 
import pickle
import streamlit as  st

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def resume_cleaning(text):
    text = re.sub("http[s]?\://\S+","",text)       # removing "http"
    text = re.sub(r"#\S+", "",text)                # removing "#"
    text = re.sub(r"@\S+", "",text)                # removing "@"
    text = re.sub(r"(\(.*\))|(\[.*\])", "",text)   # renivubg "."
    text = re.sub(r"\n", "",text)                  # removing \n
    text = re.sub('\s+',' ',text)                  # removing extra spaces
    return text


def main():
    st.title('Resume Based Job Recommendation')
    upload_file = st.file_uploader("Please Upload Your Resume", type=(['txt', 'pdf']))
    submit_button = st.button('Submit')

    if submit_button and upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        
        clean_resume = resume_cleaning(resume_text)
        features = tfidf.transform([clean_resume])
        prediction = clf.predict(features)[0]

        category_name = {   21: 'Data Science',
                            31: 'HR',
                            7: 'Advocate',
                            8: 'Arts',
                            47: 'Web Designing',
                            36: 'Mechanical Engineer',
                            44: 'Sales',
                            33: 'Health and fitness',
                            18: 'Civil Engineer',
                            35: 'Java Developer',
                            14: 'Business Analyst',
                            43: 'SAP Developer',
                            9: 'Automation Testing',
                            27: 'Electrical Engineering',
                            38: 'Operations Manager',
                            41: 'Python Developer',
                            23: 'DevOps Engineer',
                            37: 'Network Security Engineer',
                            39: 'PMO',
                            22: 'Database',
                            32: 'Hadoop',
                            26: 'ETL Developer',
                            24: 'DotNet Developer',
                            13: 'Blockchain',
                            46: 'Testing',
                            19: 'DESIGNER',
                            34: 'INFORMATION-TECHNOLOGY',
                            45: 'TEACHER',
                            1: 'ADVOCATE',
                            12: 'BUSINESS-DEVELOPMENT',
                            30: 'HEALTHCARE',
                            29: 'FITNESS',
                            2: 'AGRICULTURE',
                            11: 'BPO',
                            42: 'SALES',
                            17: 'CONSULTANT',
                            20: 'DIGITAL-MEDIA',
                            5: 'AUTOMOBILE',
                            15: 'CHEF',
                            28: 'FINANCE',
                            3: 'APPAREL',
                            25: 'ENGINEERING',
                            0: 'ACCOUNTANT',
                            16: 'CONSTRUCTION',
                            40: 'PUBLIC-RELATIONS',
                            10: 'BANKING',
                            4: 'ARTS',
                            6: 'AVIATION'
                            }

        
        
        st.write('Your Resume is eligible for:- ',category_name.get(prediction))   #category_name.get(prediction)

if __name__=="__main__":
        main()


# import streamlit as st
# import pickle
# import re
# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')

# #loading models
# clf = pickle.load(open('clf.pkl','rb'))
# tfidfd = pickle.load(open('tfidf.pkl','rb'))

# def clean_resume(resume_text):
#     clean_text = re.sub('http\S+\s*', ' ', resume_text)
#     clean_text = re.sub('RT|cc', ' ', clean_text)
#     clean_text = re.sub('#\S+', '', clean_text)
#     clean_text = re.sub('@\S+', '  ', clean_text)
#     clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
#     clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
#     clean_text = re.sub('\s+', ' ', clean_text)
#     return clean_text
# # web app
# def main():
#     st.title("Resume Screening App")
#     uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

#     if uploaded_file is not None:
#         try:
#             resume_bytes = uploaded_file.read()
#             resume_text = resume_bytes.decode('utf-8')
#         except UnicodeDecodeError:
#             # If UTF-8 decoding fails, try decoding with 'latin-1'
#             resume_text = resume_bytes.decode('latin-1')

#         cleaned_resume = clean_resume(resume_text)
#         input_features = tfidfd.transform([cleaned_resume])
#         prediction_id = clf.predict(input_features)[0]
#         st.write(prediction_id)

#         # Map category ID to category name
#         category_mapping = {
#             15: "Java Developer",
#             23: "Testing",
#             8: "DevOps Engineer",
#             20: "Python Developer",
#             24: "Web Designing",
#             12: "HR",
#             13: "Hadoop",
#             3: "Blockchain",
#             10: "ETL Developer",
#             18: "Operations Manager",
#             6: "Data Science",
#             22: "Sales",
#             16: "Mechanical Engineer",
#             1: "Arts",
#             7: "Database",
#             11: "Electrical Engineering",
#             14: "Health and fitness",
#             19: "PMO",
#             4: "Business Analyst",
#             9: "DotNet Developer",
#             2: "Automation Testing",
#             17: "Network Security Engineer",
#             21: "SAP Developer",
#             5: "Civil Engineer",
#             0: "Advocate",
#         }

#         category_name = category_mapping.get(prediction_id, "Unknown")

#         st.write("Predicted Category:", category_name)



# # python main
# if __name__ == "__main__":
#     main()