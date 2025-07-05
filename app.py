from flask import Flask, render_template, request
from faq_utils import find_best_faq_answer, ask_ollama

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    matched_question = ""
    similarity = 0
    user_question = ""

    if request.method == "POST":
        user_question = request.form["question"]
        matched_question, matched_answer, similarity = find_best_faq_answer(user_question)

        if similarity > 1.0: #not a good match
            response = ask_ollama(user_question, "")
        else :
            response = ask_ollama(user_question, matched_answer)

    return render_template("index.html",
                            response=response,
                            question=user_question,
                            match=matched_question,
                            similarity=similarity)

if __name__ == "__main__":
    app.run(debug=True)