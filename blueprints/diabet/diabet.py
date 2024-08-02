from flask import Blueprint, request, jsonify, render_template


diabetes_bp = Blueprint('diabet', __name__, template_folder='templates')

@diabetes_bp.route('/')
def index():
    return render_template('index.html')
