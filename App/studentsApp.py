import gradio as gr
import skops.io as sio
import pandas as pd

unknownTypes = sio.get_untrusted_types(file="./Model/studentsPipeline.skops")
pipe = sio.load("./Model/studentsPipeline.skops", trusted=unknownTypes)
unknownTypes = sio.get_untrusted_types(file="./Model/studentsTargetLabelEncoder.skops")
aLabelEncoder = sio.load("./Model/studentsTargetLabelEncoder.skops", trusted=unknownTypes)

def predictStudentState(
    curricularUnits2ndSemApproved,
    curricularUnits1stSemApproved,
    curricularUnits2ndSemGrade,
    curricularUnits1stSemGrade,
    curricularUnits1stSemEvaluations,
    tuitionFeesUpToDate,
    curricularUnits2ndSemEvaluations,
    course,
    gender,
    scholarshipHolder,
    applicationMode,
    ageAtEnrollment,
    curricularUnits2ndSemEnrolled,
    international,
    admissionGrade,
    curricularUnits1stSemEnrolled,
    curricularUnits2ndSemCredited,
    previousQualificationGrade,
    mothersOccupation,
    fathersQualification,
    curricularUnits1stSemCredited,
    daytimeEveningAttendance,
    mothersQualification,
    fathersOccupation,
    inflationRate,
    previousQualification,
):
    """Predict predict students' dropout and academic sucess based on student features.

    Args:
        See data source for details:
        https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

    Returns:
        str: Predicted student status
    """
    someFeatures = [
        curricularUnits2ndSemApproved,
        curricularUnits1stSemApproved,
        curricularUnits2ndSemGrade,
        curricularUnits1stSemGrade,
        curricularUnits1stSemEvaluations,
        tuitionFeesUpToDate,
        curricularUnits2ndSemEvaluations,
        course,
        gender,
        scholarshipHolder,
        applicationMode,
        ageAtEnrollment,
        curricularUnits2ndSemEnrolled,
        international,
        admissionGrade,
        curricularUnits1stSemEnrolled,
        curricularUnits2ndSemCredited,
        previousQualificationGrade,
        mothersOccupation,
        fathersQualification,
        curricularUnits1stSemCredited,
        daytimeEveningAttendance,
        mothersQualification,
        fathersOccupation,
        inflationRate,
        previousQualification,
    ]
    columnsNames = ['Curricular units 2nd sem (approved)',
       'Curricular units 1st sem (approved)',
       'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (evaluations)', 'Tuition fees up to date',
       'Curricular units 2nd sem (evaluations)', 'Course', 'Gender',
       'Scholarship holder', 'Application mode', 'Age at enrollment',
       'Curricular units 2nd sem (enrolled)', 'International',
       'Admission grade', 'Curricular units 1st sem (enrolled)',
       'Curricular units 2nd sem (credited)', 'Previous qualification (grade)',
       "Mother's occupation", "Father's qualification",
       'Curricular units 1st sem (credited)', 'Daytime/evening attendance\t',
       "Mother's qualification", "Father's occupation", 'Inflation rate',
       'Previous qualification']
    someFeatures = pd.DataFrame([someFeatures], columns=columnsNames)
    predictedStudentState = aLabelEncoder.inverse_transform(pipe.predict(someFeatures) - 1)
    print(predictedStudentState, pipe.predict_proba(someFeatures)[:, 0])

    resultsLabel = f"Predicted student state: {predictedStudentState}, \
                        \ndropout prob. : {pipe.predict_proba(someFeatures)[:, 0]}, \
                        \nenrollment prob. : {pipe.predict_proba(someFeatures)[:, 1]}, \
                        \ngraduate prob. : {pipe.predict_proba(someFeatures)[:, 2]}"
    return resultsLabel

# Variables options
courseOptions = [('Biofuel Production Technologies', 33),
 ('Animation and Multimedia Design', 171),
 ('Social Service (evening attendance)', 8014),
 ('Agronomy', 9003),
 ('Communication Design', 9070),
 ('Veterinary Nursing', 9085),
 ('Informatics Engineering', 9119),
 ('Equinculture', 9130),
 ('Management', 9147),
 ('Social Service', 9238),
 ('Tourism', 9254),
 ('Nursing', 9500),
 ('Oral Hygiene', 9556),
 ('Advertising and Marketing Management', 9670),
 ('Journalism and Communication', 9773),
 ('Basic Education', 9853),
 ('Management (evening attendance)', 9991)]
applicationModeOptions = [('1st phase - general contingent', 1),
 ('Ordinance No. 612/93', 2),
 ('1st phase - special contingent     (Azores Island)', 5),
 ('Holders of other higher courses', 7),
 ('Ordinance No. 854-B/99', 10),
 ('International student         (bachelor)', 15),
 ('1st phase - special contingent (Madeira Island)', 16),
 ('2nd phase - general contingent', 17),
 ('3rd phase - general contingent', 18),
 ('Ordinance No. 533-A/99, item b2) (Different Plan)', 26),
 ('Ordinance No. 533-A/99, item b3 (Other Institution)', 27),
 ('Over 23 years old', 39),
 ('Transfer', 42),
 ('Change of course', 43),
 ('Technological specialization diploma holders', 44),
 ('Change of institution/course', 51),
 ('Short cycle diploma holders', 53),
 ('Change of institution/course (International)',
  57)]
mothersOccupationOptions = [('Student', 0),
 ('Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
  1),
 ('Specialists in Intellectual and Scientific Activities', 2),
 ('Intermediate Level Technicians and Professions', 3),
 ('Administrative staff', 4),
 ('Personal Services, Security and Safety Workers and Sellers', 5),
 ('Farmers and Skilled Workers in Agriculture, Fisheries and Forestry', 6),
 ('Skilled Workers in Industry, Construction and Craftsmen',7),
 ('Installation and Machine Operators and Assembly Workers', 8),
 ('Unskilled Workers', 9),
 ('Armed Forces Professions', 10),
 ('Other Situation', 90),
 ('(blank)', 99),
 ('Health professionals', 122),
 ('teachers', 123),
 ('Specialists in information and communication technologies (ICT)', 125),
 ('Intermediate level science and engineering technicians and professions', 131),
 ('Technicians and professionals, of intermediate level of health', 132),
 ('Intermediate level technicians from legal, social, sports, cultural and similar services', 134),
 ('Office workers, ecretaries in general and data processing operators', 141),
 ('Data, accounting,statistical, financial services and registry-related operators', 143),
 ('Other administrative support staff', 144),
 ('personal service workers', 151),
 ('sellers', 152),
 ('Personal care workers and the like', 153),
 ('Skilled construction workers and the like, except electricians',  171),
 ('Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',  173),
 ('Workers in food processing, woodworking, clothing and other industries and crafts',  175),
 ('cleaning workers', 191),
 ('Unskilled workers in agriculture, animal production, fisheries and     forestry',  192),
 ('Unskilled workers in extractive industry, construction, manufacturing and transport',  193),
 ('Meal preparation assistants', 194)]
fathersQualificationOptions = [('Secondary Education - 12th Year of Schooling or Eq.', 1),
 ("Higher Education - Bachelor's Degree", 2),
 ('Higher Education - Degree', 3),
 ("Higher Education - Master's", 4),
 ('Higher Education - Doctorate', 5),
 ('Frequency of Higher Education', 6),
 ('12th Year of Schooling - Not Completed', 9),
 ('11th Year of Schooling - Not Completed', 10),
 ('7th Year (Old)', 11),
 ('Other - 11th Year of Schooling', 12),
 ('2nd year complementary high school course', 13),
 ('10th Year of Schooling', 14),
 ('General commerce course', 18),
 ('Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.', 19),
 ('Complementary High School Course', 20),
 ('Technical-professional course', 22),
 ('Complementary High School Course - not concluded', 25),
 ('7th year of schooling', 26),
 ('2nd cycle of the general high school course', 27),
 ('9th Year of Schooling - Not Completed', 29),
 ('8th year of schooling', 30),
 ('General Course of Administration and Commerce', 31),
 ('Supplementary Accounting and Administration', 33),
 ('Unknown', 34),
 ("Can't read or write", 35),
 ('Can read without having a 4th year of schooling', 36),
 ('Basic education 1st cycle (4th/5th year) or equiv.', 37),
 ('Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.', 38),
 ('Technological specialization course', 39),
 ('Higher education - degree (1st cycle)', 40),
 ('Specialized higher studies course', 41),
 ('Professional higher technical course', 42),
 ('Higher Education - Master (2nd cycle)', 43),
 ('Higher Education - Doctorate (3rd cycle)', 44)]
mothersQualificationOptions = [('Secondary Education - 12th Year of Schooling or Eq.', 1),
 ("Higher Education - Bachelor's Degree", 2),
 ('Higher Education - Degree', 3),
 ("Higher Education - Master's", 4),
 ('Higher Education - Doctorate', 5),
 ('Frequency of Higher Education', 6),
 ('12th Year of Schooling - Not Completed', 9),
 ('11th Year of Schooling - Not Completed', 10),
 ('7th Year (Old)', 11),
 ('Other - 11th Year of Schooling', 12),
 ('10th Year of Schooling', 14),
 ('General commerce course', 18),
 ('Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.', 19),
 ('Technical-professional course', 22),
 ('7th year of schooling', 26),
 ('2nd cycle of the general high school course', 27),
 ('9th Year of Schooling - Not Completed', 29),
 ('8th year of schooling', 30),
 ('Unknown', 34),
 ("Can't read or write", 35),
 ('Can read without having a 4th year of schooling', 36),
 ('Basic education 1st cycle (4th/5th year) or equiv.', 37),
 ('Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.', 38),
 ('Technological specialization course', 39),
 ('Higher education - degree (1st cycle)', 40),
 ('Specialized higher studies course', 41),
 ('Professional higher technical course', 42),
 ('Higher Education - Master (2nd cycle)', 43),
 ('Higher Education - Doctorate (3rd cycle)', 44)]
fathersOccupationOptions = [('Student', 0),
 ('Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
  1),
 ('Specialists in Intellectual and Scientific Activities', 2),
 ('Intermediate Level Technicians and Professions', 3),
 ('Administrative staff', 4),
 ('Personal Services, Security and Safety Workers and Sellers', 5),
 ('Farmers and Skilled Workers in Agriculture, Fisheries and Forestry', 6),
 ('Skilled Workers in Industry, Construction and Craftsmen', 7),
 ('Installation and Machine Operators and Assembly Workers', 8),
 ('Unskilled Workers', 9),
 ('Armed Forces Professions', 10),
 ('Other Situation', 90),
 ('(blank)', 99),
 ('Armed Forces Officers', 101),
 ('Armed Forces Sergeants', 102),
 ('Other Armed Forces personnel', 103),
 ('Directors of administrative and commercial services', 112),
 ('Hotel, catering, trade and other services directors', 114),
 ('Specialists in the physical sciences, mathematics, engineering and related techniques',
  121),
 ('Health professionals', 122),
 ('teachers', 123),
 ('Specialists in finance, accounting, administrative organization, public and commercial relations',
  124),
 ('Intermediate level science and engineering technicians and professions',
  131),
 ('Technicians and professionals, of intermediate level of health', 132),
 ('Intermediate level technicians from legal, social, sports, cultural and similar services',
  134),
 ('Information and communication technology technicians', 135),
 ('Office workers, secretaries in general and data processing operators', 141),
 ('Data, accounting, statistical, financial services and registry-related operators',
  143),
 ('Other administrative support staff', 144),
 ('personal service workers', 151),
 ('sellers', 152),
 ('Personal care workers and the like', 153),
 ('Protection and security services personnel', 154),
 ('Market-oriented farmers and skilled agricultural and animal production workers',
  161),
 ('Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence',
  163),
 ('Skilled construction workers and the like, except electricians', 171),
 ('Skilled workers in metallurgy, metalworking and similar', 172),
 ('Skilled workers in electricity and electronics', 174),
 ('Workers in food processing, woodworking, clothing and other industries and crafts',
  175),
 ('Fixed plant and machine operators', 181),
 ('assembly workers', 182),
 ('Vehicle drivers and mobile equipment operators', 183),
 ('Unskilled workers in agriculture, animal production, fisheries and forestry',
  192),
 ('Unskilled workers in extractive industry, construction, manufacturing and transport',
  193),
 ('Meal preparation assistants', 194),
 ('Street vendors (except food) and street service providers', 195)]
previousQualificationOptions = [('Secondary education', 1),
 ("Higher education - bachelor's degree", 2),
 ('Higher education - degree', 3),
 ("Higher education - master's", 4),
 ('Higher education - doctorate', 5),
 ('Frequency of higher education', 6),
 ('12th year of schooling - not completed', 9),
 ('11th year of schooling - not completed', 10),
 ('Other - 11th year of schooling', 12),
 ('10th year of schooling', 14),
 ('10th year of schooling - not completed', 15),
 ('Basic education 3rd cycle (9th/10th/11th year) or equiv.', 19),
 ('Basic education 2nd cycle (6th/7th/8th year) or equiv.', 38),
 ('Technological specialization course', 39),
 ('Higher education - degree (1st cycle)', 40),
 ('Professional higher technical course', 42),
 ('Higher education - master (2nd cycle)', 43)]
appInputs = [
    # Curricular units 2nd sem (approved)
    gr.Slider(0, 40, step=1, label='Curricular units 2nd sem (approved)'),
    # Curricular units 2nd sem (approved)
    gr.Slider(0, 40, step=1, label='Curricular units 1st sem (approved)'),
    # Curricular units 2nd sem (grade)
    gr.Slider(0, 30, step=0.1, label='Curricular units 2nd sem (grade)'),
    # Curricular units 1st sem (grade)
    gr.Slider(0, 30, step=0.1, label='Curricular units 1st sem (grade)'),
    # Curricular units 1st sem (evaluations)
    gr.Slider(0, 50, step=1, label='Curricular units 1st sem (evaluations)'),
    # Tuition fees up to date
    gr.Radio([('Yes', 1), ('No', 0)], label='Tuition fees up to date'),
    # Curricular units 2nd sem (evaluations)
    gr.Slider(0, 50, step=1, label='Curricular units 2nd sem (evaluations)'),
    # Course
    gr.Dropdown(courseOptions, multiselect=False , label='Course'),
    # Gender
    gr.Radio([('Male', 1), ('Female', 0)], label='Gender'),
    # Scholarship holder
    gr.Radio([('Yes', 1), ('No', 0)], label='Scholarship holder'),    
    # Application mode
    gr.Dropdown(applicationModeOptions, multiselect=False , label='Application mode'),
    # Age at enrollment
    gr.Slider(12, 99, step=1, label='Age at enrollment'),
    # Curricular units 2nd sem (enrolled)
    gr.Slider(0, 40, step=1, label='Curricular units 2nd sem (enrolled)'),
    # International
    gr.Radio([('Yes', 1), ('No', 0)], label='International'),
    # Admission grade
    gr.Slider(0, 200, step=1, label='Admission grade'),
    # Curricular units 1st sem (enrolled)
    gr.Slider(0, 40, step=1, label='Curricular units 1st sem (enrolled)'),
    # Curricular units 2nd sem (credited)
    gr.Slider(0, 40, step=1, label='Curricular units 2nd sem (credited)'),    
    # Previous qualification (grade)
    gr.Slider(0, 200, step=1, label='Previous qualification (grade)'),
    # Mother's occupation
    gr.Dropdown(mothersOccupationOptions, multiselect=False , label="Mother's occupation"),    
    # Father's qualification
    gr.Dropdown(fathersQualificationOptions, multiselect=False , label="Father's qualification"),  
    # Curricular units 1st sem (credited)
    gr.Slider(0, 40, step=1, label='Curricular units 1st sem (credited)'), 
    # Daytime/evening attendance
    gr.Radio([('Daytime', 1), ('Evening', 0)], label='Daytime/evening attendance'),
    # Mother's qualification
    gr.Dropdown(mothersQualificationOptions, multiselect=False , label="Mother's qualification"),
    # Fathers's occupation
    gr.Dropdown(fathersOccupationOptions, multiselect=False , label="Father's occupation"),      
    # Inflation rate
    gr.Slider(-2, 5, step=0.1, label='Inflation rate'), 
    # Previous qualification
    gr.Dropdown(previousQualificationOptions, multiselect=False , label="Previous qualification")
]

appOutputs = [gr.Label(num_top_classes=3)]

# someExamples = pd.read_csv('./Model/examples.csv')


aTitle = "Student's state prediction"
aDescription = "Enter the details to predict students' dropout and academic sucess"
anArticle = "This app uses CI/CD practices for machine learning"

gr.Interface(
    fn=predictStudentState,
    inputs=appInputs,
    outputs=appOutputs,
    # examples=someExamples,
    title=aTitle,
    description=aDescription,
    article=anArticle,
    theme=gr.themes.Citrus()
).launch()