from flask import Flask, request, jsonify
import cirq
import numpy as np
from werkzeug.exceptions import BadRequest
from flask_cors import CORS
import json
import time
from collections import defaultdict
import openai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import qutip as qt
from scipy import linalg
import matplotlib.pyplot as plt
import io
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv('env')

class QuantumStateVisualizer:
    @staticmethod
    def state_to_bloch(state_vector: np.ndarray) -> str:
        """Convert quantum state to Bloch sphere visualization"""
        # Convert state vector to density matrix
        rho = np.outer(state_vector, state_vector.conj())
        
        # Calculate Bloch vector components
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])
        
        x = np.real(np.trace(rho @ sx))
        y = np.real(np.trace(rho @ sy))
        z = np.real(np.trace(rho @ sz))
        
        # Create Bloch sphere visualization
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='b', alpha=0.1)
        
        # Plot state vector
        ax.quiver(0, 0, 0, x, y, z, length=1.0, color='r', arrow_length_ratio=0.2)
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

class CircuitOptimizer:
    @staticmethod
    def optimize_circuit(circuit: cirq.Circuit) -> cirq.Circuit:
        """Optimize quantum circuit using various techniques"""
        # Merge adjacent single-qubit gates
        circuit = cirq.merge_single_qubit_gates_into_phased_x_z(circuit)
        
        # Remove redundant gates
        circuit = cirq.drop_empty_moments(circuit)
        
        # Decompose multi-qubit gates
        circuit = cirq.decompose(circuit)
        
        return circuit

class NoiseSimulator:
    def __init__(self):
        self.noise_models = {
            'depolarizing': cirq.depolarize,
            'amplitude_damping': cirq.amplitude_damp,
            'phase_damping': cirq.phase_damp,
            'bit_flip': cirq.bit_flip,
            'phase_flip': cirq.phase_flip,
        }
    
    def add_noise(self, circuit: cirq.Circuit, noise_type: str, noise_level: float) -> cirq.Circuit:
        """Add noise to the circuit"""
        if noise_type not in self.noise_models:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noise_model = self.noise_models[noise_type](noise_level)
        return circuit.with_noise(noise_model)

class QuantumErrorCorrection:
    @staticmethod
    def encode_state(state: np.ndarray, code: str = 'steane') -> np.ndarray:
        """Encode quantum state using error correction code"""
        if code == 'steane':
            # Implement Steane code encoding
            pass
        elif code == 'shor':
            # Implement Shor code encoding
            pass
        return state

class QuantumAlgorithms:
    @staticmethod
    def grover_search(qubits, oracle_function, iterations=None):
        """Implement Grover's search algorithm"""
        n = len(qubits)
        if iterations is None:
            iterations = int(np.pi/4 * np.sqrt(2**n))
        
        circuit = cirq.Circuit()
        
        # Initialize superposition
        circuit.append(cirq.H.on_each(*qubits))
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle
            oracle_function(circuit, qubits)
            # Diffusion operator
            circuit.append(cirq.H.on_each(*qubits))
            circuit.append(cirq.X.on_each(*qubits))
            circuit.append(cirq.H.on(qubits[-1]))
            circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
            circuit.append(cirq.H.on(qubits[-1]))
            circuit.append(cirq.X.on_each(*qubits))
            circuit.append(cirq.H.on_each(*qubits))
        
        return circuit

    @staticmethod
    def quantum_fourier_transform(qubits):
        """Implement Quantum Fourier Transform"""
        circuit = cirq.Circuit()
        n = len(qubits)
        
        for i in range(n):
            circuit.append(cirq.H(qubits[i]))
            for j in range(i + 1, n):
                circuit.append((cirq.CZ ** (1/(2 ** (j-i))))(qubits[i], qubits[j]))
        
        # Swap qubits to complete QFT
        for i in range(n//2):
            circuit.append(cirq.SWAP(qubits[i], qubits[n-1-i]))
        
        return circuit

    @staticmethod
    def quantum_phase_estimation(qubits, unitary_operation, precision_qubits):
        """Implement Quantum Phase Estimation"""
        circuit = cirq.Circuit()
        n = len(qubits)
        
        # Initialize precision qubits in superposition
        circuit.append(cirq.H.on_each(*precision_qubits))
        
        # Controlled unitary operations
        for i, qubit in enumerate(precision_qubits):
            for _ in range(2**i):
                circuit.append(unitary_operation.controlled_by(qubit))
        
        # Inverse QFT
        circuit.append(QuantumAlgorithms.inverse_qft(precision_qubits))
        
        return circuit

    @staticmethod
    def inverse_qft(qubits):
        """Implement Inverse Quantum Fourier Transform"""
        circuit = cirq.Circuit()
        n = len(qubits)
        
        # Swap qubits first
        for i in range(n//2):
            circuit.append(cirq.SWAP(qubits[i], qubits[n-1-i]))
        
        for i in range(n-1, -1, -1):
            for j in range(i-1, -1, -1):
                circuit.append((cirq.CZ ** (-1/(2 ** (i-j))))(qubits[j], qubits[i]))
            circuit.append(cirq.H(qubits[i]))
        
        return circuit

class AdvancedNoiseSimulator(NoiseSimulator):
    def __init__(self):
        super().__init__()
        self.noise_models.update({
            'kraus': self._kraus_noise,
            'custom': self._custom_noise,
            'cross_talk': self._cross_talk_noise
        })
    
    def _kraus_noise(self, circuit: cirq.Circuit, kraus_operators: List[np.ndarray]) -> cirq.Circuit:
        """Apply Kraus operator noise model"""
        def kraus_channel(qubits):
            return cirq.KrausChannel(kraus_operators).on_each(*qubits)
        return circuit.with_noise(kraus_channel)
    
    def _cross_talk_noise(self, circuit: cirq.Circuit, crosstalk_strength: float) -> cirq.Circuit:
        """Apply crosstalk noise between adjacent qubits"""
        def crosstalk_channel(qubits):
            return cirq.depolarize(crosstalk_strength).on_each(*qubits)
        return circuit.with_noise(crosstalk_channel)
    
    def _custom_noise(self, circuit: cirq.Circuit, noise_function) -> cirq.Circuit:
        """Apply custom noise model defined by user"""
        return circuit.with_noise(noise_function)

class QuantumStateTomography:
    @staticmethod
    def measure_state(state_vector: np.ndarray, num_measurements: int = 1000) -> Dict[str, float]:
        """Perform quantum state tomography"""
        measurements = defaultdict(int)
        probabilities = np.abs(state_vector) ** 2
        
        for _ in range(num_measurements):
            outcome = np.random.choice(len(state_vector), p=probabilities)
            measurements[format(outcome, f'0{int(np.log2(len(state_vector)))}b')] += 1
        
        return {k: v/num_measurements for k, v in measurements.items()}

class QuantumMachineLearning:
    @staticmethod
    def quantum_kernel_estimation(circuit: cirq.Circuit, data_points: List[np.ndarray]) -> np.ndarray:
        """Estimate quantum kernel for machine learning"""
        n = len(data_points)
        kernel_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1):
                # Create parameterized circuit for each data point
                param_circuit = circuit.with_parameters(data_points[i])
                param_circuit2 = circuit.with_parameters(data_points[j])
                
                # Measure overlap between states
                kernel_matrix[i,j] = kernel_matrix[j,i] = abs(
                    np.vdot(
                        param_circuit.final_state_vector(),
                        param_circuit2.final_state_vector()
                    )
                ) ** 2
        
        return kernel_matrix

class EnhancedCirqSimulator:
    def __init__(self):
        self.simulator = cirq.Simulator()
        self.circuit_cache = {}
        self.result_cache = {}
        self.noise_simulator = AdvancedNoiseSimulator()
        self.visualizer = QuantumStateVisualizer()
        self.optimizer = CircuitOptimizer()
        self.algorithms = QuantumAlgorithms()
        self.tomography = QuantumStateTomography()
        self.qml = QuantumMachineLearning()
    
    def _generate_circuit_id(self, circuit_json):
        """Generate a unique ID for the circuit based on its content"""
        return hash(json.dumps(circuit_json, sort_keys=True))
    
    def run_experiment(self, circuit_json, noise_config=None, optimization_level=0):
        try:
            circuit_id = self._generate_circuit_id(circuit_json)
            
            # Check cache first
            if circuit_id in self.result_cache:
                return self.result_cache[circuit_id]
                
            circuit = self._deserialize_circuit(circuit_json)
            
            # Apply optimizations if requested
            if optimization_level > 0:
                circuit = self.optimizer.optimize_circuit(circuit)
            
            # Add noise if configured
            if noise_config:
                circuit = self.noise_simulator.add_noise(
                    circuit,
                    noise_config['type'],
                    noise_config['level']
                )
            
            start_time = time.time()
            result = self.simulator.simulate(circuit)
            execution_time = time.time() - start_time
            
            # Convert results to serializable format
            processed_result = self._process_result(result)
            processed_result['execution_time'] = execution_time
            processed_result['qubit_count'] = len(circuit.all_qubits())
            processed_result['gate_count'] = len(list(circuit.all_operations()))
            
            # Add visualization if requested
            if circuit_json.get('visualize', False):
                processed_result['visualization'] = self.visualizer.state_to_bloch(
                    result.state_vector()
                )
            
            # Cache the result
            self.result_cache[circuit_id] = processed_result
            return processed_result

        except Exception as e:
            raise ValueError(f"Error simulating circuit: {str(e)}")
    
    def _process_result(self, result):
        """Convert cirq result to serializable format"""
        state_vector = []
        for amplitude in result.state_vector():
            state_vector.append({
                'real': float(amplitude.real),
                'imag': float(amplitude.imag)
            })

        measurements = defaultdict(list)
        if hasattr(result, 'measurements'):
            for key, value in result.measurements.items():
                measurements[key] = value.tolist()

        return {
            'state_vector': state_vector,
            'measurements': dict(measurements)
        }
    
    def _qft(self, qubits):
        """Quantum Fourier Transform on the given qubits."""
        qreg = list(qubits)
        for i in range(len(qreg)):
            yield cirq.H(qreg[i])
            for j in range(1, len(qreg) - i):
                yield (cirq.CZ ** (1/(2 ** j)))(qreg[i], qreg[i + j])
                yield cirq.rz(np.pi/(2 ** j)).on(qreg[i + j])
        # Swap the qubits to complete QFT
        for i in range(len(qreg) // 2):
            yield cirq.SWAP(qreg[i], qreg[len(qreg) - i - 1])
    
    def _deserialize_circuit(self, circuit_json):
        """Convert JSON circuit description to Cirq circuit"""
        circuit_id = self._generate_circuit_id(circuit_json)
        
        # Check cache first
        if circuit_id in self.circuit_cache:
            return self.circuit_cache[circuit_id]
            
        qubits = [cirq.GridQubit(*q) for q in circuit_json.get('qubits', [[0, i] for i in range(len(circuit_json.get('operations', [])))])]
        circuit = cirq.Circuit()

        for op in circuit_json.get('operations', []):
            gate_type = op['gate']
            targets = [qubits[i] for i in op['targets']]
            controls = [qubits[i] for i in op.get('controls', [])]

            if gate_type == 'H':
                circuit.append(cirq.H.on_each(*targets))
            elif gate_type == 'X':
                if controls:
                    circuit.append(cirq.X.on(targets[0]).controlled_by(*controls))
                else:
                    circuit.append(cirq.X.on_each(*targets))
            elif gate_type == 'Y':
                if controls:
                    circuit.append(cirq.Y.on(targets[0]).controlled_by(*controls))
                else:
                    circuit.append(cirq.Y.on_each(*targets))
            elif gate_type == 'Z':
                if controls:
                    circuit.append(cirq.Z.on(targets[0]).controlled_by(*controls))
                else:
                    circuit.append(cirq.Z.on_each(*targets))
            elif gate_type == 'CNOT':
                if not controls and len(targets) < 2:
                    raise ValueError("CNOT gate requires either controls or two targets")
                control = controls[0] if controls else targets[0]
                target = targets[0] if controls else targets[1]
                if control == target:
                    raise ValueError("Control and target qubits must be different")
                circuit.append(cirq.CNOT(control, target))
            elif gate_type == 'SWAP':
                circuit.append(cirq.SWAP(targets[0], targets[1]))
            elif gate_type == 'RX':
                circuit.append(cirq.rx(op['angle']).on_each(*targets))
            elif gate_type == 'RY':
                circuit.append(cirq.ry(op['angle']).on_each(*targets))
            elif gate_type == 'RZ':
                circuit.append(cirq.rz(op['angle']).on_each(*targets))
            elif gate_type == 'RXX':
                if len(targets) < 2 and not controls:
                    raise ValueError("RXX gate requires either controls or two targets")
                control = controls[0] if controls else targets[0]
                target = targets[0] if controls else targets[1]
                if control == target:
                    raise ValueError("Control and target qubits must be different")
                circuit.append(cirq.RXX(op['angle'])(control, target))
            elif gate_type == 'RYY':
                if len(targets) < 2 and not controls:
                    raise ValueError("RYY gate requires either controls or two targets")
                control = controls[0] if controls else targets[0]
                target = targets[0] if controls else targets[1]
                if control == target:
                    raise ValueError("Control and target qubits must be different")
                circuit.append(cirq.RYY(op['angle'])(control, target))
            elif gate_type == 'RZZ':
                if len(targets) < 2 and not controls:
                    raise ValueError("RZZ gate requires either controls or two targets")
                control = controls[0] if controls else targets[0]
                target = targets[0] if controls else targets[1]
                if control == target:
                    raise ValueError("Control and target qubits must be different")
                circuit.append(cirq.RZZ(op['angle'])(control, target))
            elif gate_type == 'CCNOT':
                if len(controls) >= 2 and len(targets) >= 1:
                    # Ensure all qubits are unique
                    all_qubits = controls + targets[:1]
                    if len(set(all_qubits)) != len(all_qubits):
                        raise ValueError("CCNOT gate requires unique qubits for controls and target")
                    circuit.append(cirq.TOFFOLI(controls[0], controls[1], targets[0]))
                else:
                    raise ValueError("CCNOT/Toffoli gate requires exactly two controls and one target")
            elif gate_type == 'S':
                circuit.append(cirq.S.on_each(*targets))
            elif gate_type == 'T':
                circuit.append(cirq.T.on_each(*targets))
            elif gate_type == 'I':
                circuit.append(cirq.I.on_each(*targets))
            elif gate_type == 'CPHASE':
                circuit.append(cirq.CZ(targets[0], targets[1]))
            elif gate_type == 'Measure':
                circuit.append(cirq.measure(*targets, key=op.get('key', 'result')))
            elif gate_type == 'QFT':
                circuit.append(self._qft(targets))

        # Cache the circuit
        self.circuit_cache[circuit_id] = circuit
        return circuit

simulator = EnhancedCirqSimulator()

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.get_json()
        if not data or 'circuit' not in data:
            raise BadRequest("Circuit data is required")

        noise_config = data.get('noise_config')
        optimization_level = data.get('optimization_level', 0)
        
        result = simulator.run_experiment(
            data['circuit'],
            noise_config=noise_config,
            optimization_level=optimization_level
        )
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/quantum-assistant', methods=['POST'])
def quantum_assistant():
    try:
        if not openai.api_key:
            return jsonify({'success': False, 'error': 'OpenAI API key not configured'}), 500

        data = request.get_json()
        if not data or 'messages' not in data or 'qubit_count' not in data:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        system_msg = {
            "role": "system",
            "content": f"""You are a quantum computing assistant specialized in circuit construction. Follow these rules:
1. Always respond with a clear explanation first
2. For circuit modifications, include a JSON array after '||' with this exact format:
   [{{"action": "add_qubit"}}] or [{{"action": "remove_qubit", "index": qubit_index}}] or [{{"action": "remove_gate", "gate_id": gate_id}}] or [{{"gate": "GATE_TYPE", "targets": [qubit_index], "controls": [optional_control_indices], "angle": optional_angle}}]
   
Available gates: H, X, Y, Z, CNOT, SWAP, RX, RY, RZ, RXX, RYY, RZZ, CCNOT, S, T, CPHASE, Measure, QFT
Current qubits: {data['qubit_count']}
Qubit indices: 0 to {data['qubit_count']-1}

Example valid responses:
"To create superposition, apply Hadamard to qubit 0. || [{{\"gate\": \"H\", \"targets\": [0]}}]"
"Entangle qubits 0 and 1 with CNOT (0 controls 1) || [{{\"gate\": \"CNOT\", \"targets\": [1], \"controls\": [0]}}]"
"Rotate qubit 0 by π/2 around X-axis || [{{\"gate\": \"RX\", \"targets\": [0], \"angle\": 1.5708}}]"
"Let's add a new qubit || [{{\"action\": \"add_qubit\"}}]"
"Let's remove qubit 2 || [{{\"action\": \"remove_qubit\", \"index\": 2}}]"
"Let's remove the gate with ID 5 || [{{\"action\": \"remove_gate\", \"gate_id\": 5}}]"
"""
        }

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[system_msg] + data['messages'],
            temperature=0.3
        )

        full_response = response.choices[0].message.content
        if '||' in full_response:
            explanation, commands_str = full_response.split('||', 1)
            try:
                commands = json.loads(commands_str.strip())
                valid_commands = []
                for cmd in commands:
                    if 'action' in cmd:
                        if cmd['action'] == 'add_qubit':
                            valid_commands.append(cmd)
                        elif cmd['action'] == 'remove_qubit':
                            if isinstance(cmd.get('index'), int) and 0 <= cmd['index'] < data['qubit_count']:
                                valid_commands.append(cmd)
                        elif cmd['action'] == 'remove_gate':
                            if isinstance(cmd.get('gate_id'), int):
                                valid_commands.append(cmd)
                    else:
                        if not all(k in cmd for k in ['gate', 'targets']):
                            continue
                        if cmd['gate'] not in ['H', 'X', 'Y', 'Z', 'CNOT', 'SWAP', 'RX', 'RY', 'RZ', 'CCNOT', 'S', 'T', 'CPHASE', 'Measure', 'QFT']:
                            continue
                        if any(t >= data['qubit_count'] for t in cmd['targets']):
                            continue
                        if 'controls' in cmd and any(c >= data['qubit_count'] for c in cmd['controls']):
                            continue
                        valid_commands.append(cmd)
            except json.JSONDecodeError:
                valid_commands = []
        else:
            explanation = full_response
            valid_commands = []

        return jsonify({
            'success': True,
            'explanation': explanation.strip(),
            'commands': valid_commands
        })

    except openai.error.AuthenticationError:
        return jsonify({'success': False, 'error': 'Invalid OpenAI API key'}), 401
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gates', methods=['GET'])
def available_gates():
    gates = [
        {'name': 'H', 'description': 'Hadamard gate'},
        {'name': 'X', 'description': 'Pauli-X gate'},
        {'name': 'Y', 'description': 'Pauli-Y gate'},
        {'name': 'Z', 'description': 'Pauli-Z gate'},
        {'name': 'CNOT', 'description': 'Controlled-NOT gate'},
        {'name': 'SWAP', 'description': 'SWAP gate'},
        {'name': 'RX', 'description': 'X-axis rotation'},
        {'name': 'RY', 'description': 'Y-axis rotation'},
        {'name': 'RZ', 'description': 'Z-axis rotation'},
        {'name': 'CCNOT', 'description': 'Toffoli gate'},
        {'name': 'S', 'description': 'Phase gate'},
        {'name': 'T', 'description': 'T gate'},
        {'name': 'I', 'description': 'Identity gate'},
        {'name': 'CPHASE', 'description': 'Controlled Phase gate'},
        {'name': 'MEASURE', 'description': 'Measurement'},
        {'name': 'QFT', 'description': 'Quantum Fourier Transform'},
        {'name': 'RXX', 'description': 'XX rotation'},
        {'name': 'RYY', 'description': 'YY rotation'},
        {'name': 'RZZ', 'description': 'ZZ rotation'}
    ]
    return jsonify({'gates': gates})

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    simulator.circuit_cache.clear()
    simulator.result_cache.clear()
    return jsonify({'success': True, 'message': 'Cache cleared'})

@app.route('/api/noise-models', methods=['GET'])
def available_noise_models():
    """Return available noise models"""
    return jsonify({
        'success': True,
        'models': list(simulator.noise_simulator.noise_models.keys())
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_circuit():
    """Optimize a quantum circuit"""
    try:
        data = request.get_json()
        if not data or 'circuit' not in data:
            raise BadRequest("Circuit data is required")
        
        circuit = simulator._deserialize_circuit(data['circuit'])
        optimized_circuit = simulator.optimizer.optimize_circuit(circuit)
        
        return jsonify({
            'success': True,
            'optimized_circuit': str(optimized_circuit)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/quantum-algorithms', methods=['POST'])
def run_quantum_algorithm():
    """Endpoint for running quantum algorithms"""
    try:
        data = request.json
        algorithm_type = data.get('algorithm')
        params = data.get('params', {})
        
        simulator = EnhancedCirqSimulator()
        
        if algorithm_type == 'grover':
            qubits = [cirq.GridQubit(*q) for q in params.get('qubits', [[0, i] for i in range(3)])]
            oracle = lambda c, q: c.append(cirq.X.on_each(*q))
            circuit = simulator.algorithms.grover_search(qubits, oracle)
        elif algorithm_type == 'qft':
            qubits = [cirq.GridQubit(*q) for q in params.get('qubits', [[0, i] for i in range(3)])]
            circuit = simulator.algorithms.quantum_fourier_transform(qubits)
        elif algorithm_type == 'phase_estimation':
            qubits = [cirq.GridQubit(*q) for q in params.get('qubits', [[0, i] for i in range(3)])]
            precision_qubits = [cirq.GridQubit(*q) for q in params.get('precision_qubits', [[1, i] for i in range(2)])]
            unitary = cirq.X
            circuit = simulator.algorithms.quantum_phase_estimation(qubits, unitary, precision_qubits)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
        result = simulator.run_experiment(circuit)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/tomography', methods=['POST'])
def perform_tomography():
    """Endpoint for quantum state tomography"""
    try:
        data = request.json
        state_vector = np.array(data.get('state_vector'))
        num_measurements = data.get('num_measurements', 1000)
        
        tomography = QuantumStateTomography()
        measurements = tomography.measure_state(state_vector, num_measurements)
        
        return jsonify({'measurements': measurements})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/quantum-kernel', methods=['POST'])
def estimate_quantum_kernel():
    """Endpoint for quantum kernel estimation"""
    try:
        data = request.json
        circuit_json = data.get('circuit')
        data_points = data.get('data_points')
        
        simulator = EnhancedCirqSimulator()
        circuit = simulator._deserialize_circuit(circuit_json)
        
        kernel_matrix = simulator.qml.quantum_kernel_estimation(circuit, data_points)
        
        return jsonify({'kernel_matrix': kernel_matrix.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')