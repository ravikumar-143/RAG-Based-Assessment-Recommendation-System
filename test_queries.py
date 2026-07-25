import requests

queries = [
    "Python developer with SQL",
    "Java Developer",
    "Azure Data Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "HR Recruiter",
    "Project Manager",
    "React Developer",
    "Business Analyst",
    "DevOps Engineer"
]

for q in queries:
    print("=" * 80)
    print("QUERY:", q)

    response = requests.post(
        "http://127.0.0.1:8001/recommend",
        json={"query": q}
    )

    data = response.json()

    for i, item in enumerate(data["recommended_assessments"], start=1):
        print(i, "-", item["name"])

    print()




