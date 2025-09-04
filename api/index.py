from flask import Flask, render_template, request, jsonify
import os
import traceback
from detection import analyze_traffic_scene

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
DETECTIONS_FOLDER = os.path.join(STATIC_FOLDER, 'detections')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

for folder in [UPLOAD_FOLDER, STATIC_FOLDER, DETECTIONS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'mp4'}

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_lanes():
    """Handles the initial analysis of up to 3 lane files."""
    results = {}
    files = [request.files.get('file-1'), request.files.get('file-2'), request.files.get('file-3')]

    if not any(f and f.filename for f in files):
        return jsonify({'error': 'No files were selected for analysis.'}), 400

    for i, file in enumerate(files):
        lane_id = f"lane_{i+1}"
        if file and file.filename and allowed_file(file.filename):
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{lane_id}_{file.filename}")
                file.save(filepath)
                
                analysis_result = analyze_traffic_scene(filepath)
                
                green_time = max(10, min(analysis_result['vehicle_count'] * 2, 60))
                analysis_result['green_time'] = green_time
                
                results[lane_id] = analysis_result

            except Exception as e:
                print(traceback.format_exc())
                results[lane_id] = {'error': f'Analysis failed: {str(e)}'}
        else:
            results[lane_id] = None 
            
    return jsonify(results)

@app.route('/get_priority', methods=['POST'])
def get_priority():
    """The 'brain' of the simulation. Determines the next green light."""
    lane_data = request.json.get('lanes', {})
    active_lanes = {k: v for k, v in lane_data.items() if v and v.get('vehicle_count', 0) > 0}

    if not active_lanes:
        return jsonify({'next_lane': None, 'reason': 'No active lanes remaining.'})

    for lane_id, data in active_lanes.items():
        if data.get('has_emergency'):
            return jsonify({'next_lane': lane_id, 'reason': f'Emergency vehicle detected in {lane_id}.'})

    for lane_id, data in active_lanes.items():
        if data.get('has_accident'):
            return jsonify({'next_lane': lane_id, 'reason': f'Accident detected in {lane_id}.'})

    next_lane = max(active_lanes, key=lambda k: active_lanes[k]['vehicle_count'])
    reason = f'Highest vehicle count ({active_lanes[next_lane]["vehicle_count"]}) in {next_lane}.'
    
    return jsonify({'next_lane': next_lane, 'reason': reason})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, reloader_type='stat')
