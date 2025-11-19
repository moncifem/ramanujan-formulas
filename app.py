from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import asyncio
import threading
import json
from datetime import datetime
import logging
from config import Config
from swarm_orchestrator import RamanujanSwarm
from oeis_verifier import OEISVerifier
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.FLASK_SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global instances
swarm = None
verifier = OEISVerifier()
discovery_thread = None

def create_swarm_callback():
    """Create callback function for swarm updates"""
    def callback(event_type, data):
        try:
            socketio.emit('swarm_update', {
                'event_type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    return callback

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/start_discovery', methods=['POST'])
def start_discovery():
    """Start mathematical discovery session"""
    global swarm, discovery_thread
    
    try:
        data = request.get_json()
        target_constant = data.get('target_constant', 'pi')
        max_generations = data.get('max_generations', 50)
        
        if swarm and swarm.state.running:
            return jsonify({'error': 'Discovery session already running'}), 400
        
        # Initialize swarm
        swarm = RamanujanSwarm()
        swarm.add_callback(create_swarm_callback())
        
        # Start discovery in background thread
        def run_discovery():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    swarm.run_discovery_session(target_constant, max_generations)
                )
            except Exception as e:
                logger.error(f"Discovery session error: {e}")
                socketio.emit('error', {'message': str(e)})
            finally:
                loop.close()
        
        discovery_thread = threading.Thread(target=run_discovery)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        return jsonify({
            'status': 'started',
            'target_constant': target_constant,
            'max_generations': max_generations
        })
        
    except Exception as e:
        logger.error(f"Start discovery error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_discovery', methods=['POST'])
def stop_discovery():
    """Stop mathematical discovery session"""
    global swarm
    
    try:
        if swarm:
            swarm.stop_discovery()
            return jsonify({'status': 'stopped'})
        else:
            return jsonify({'error': 'No active discovery session'}), 400
            
    except Exception as e:
        logger.error(f"Stop discovery error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get current discovery status"""
    global swarm
    
    try:
        if swarm:
            state = swarm.get_current_state()
            return jsonify(state)
        else:
            return jsonify({
                'generation': 0,
                'target_constant': None,
                'running': False,
                'discoveries': [],
                'total_evaluated': 0
            })
            
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify_discovery', methods=['POST'])
def verify_discovery():
    """Verify a discovery against OEIS"""
    try:
        data = request.get_json()
        expression = data.get('expression')
        value = data.get('value')
        error = data.get('error')
        
        if not all([expression, value is not None, error is not None]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        verification = verifier.verify_discovery(expression, value, error)
        return jsonify(verification)
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/discoveries')
def get_discoveries():
    """Get all discoveries"""
    global swarm
    
    try:
        if swarm and swarm.discoveries:
            return jsonify({'discoveries': swarm.discoveries})
        else:
            return jsonify({'discoveries': []})
            
    except Exception as e:
        logger.error(f"Get discoveries error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_discoveries')
def export_discoveries():
    """Export discoveries to JSON"""
    global swarm
    
    try:
        if swarm and swarm.discoveries:
            # Add verification for novel discoveries
            verified_discoveries = verifier.batch_verify_discoveries(swarm.discoveries)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_discoveries': len(verified_discoveries),
                'target_constant': swarm.state.target_constant,
                'generations_run': swarm.state.generation,
                'total_expressions_evaluated': swarm.state.total_expressions_evaluated,
                'discoveries': verified_discoveries
            }
            
            return jsonify(export_data)
        else:
            return jsonify({'error': 'No discoveries to export'}), 400
            
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('connected', {'message': 'Connected to Ramanujan-Swarm'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    global swarm
    
    try:
        if swarm:
            state = swarm.get_current_state()
            emit('status_update', state)
        else:
            emit('status_update', {
                'generation': 0,
                'target_constant': None,
                'running': False,
                'discoveries': [],
                'total_evaluated': 0
            })
    except Exception as e:
        logger.error(f"Status request error: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    # Check for required environment variables
    if not Config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set. Please set your API key in environment variables.")
        exit(1)
    
    logger.info("Starting Ramanujan-Swarm Web Interface...")
    logger.info(f"Target constants available: {list(Config.TARGET_CONSTANTS.keys())}")
    
    # Run the app
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG,
        allow_unsafe_werkzeug=True
    )
