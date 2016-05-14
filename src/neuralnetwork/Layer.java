package neuralnetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Artem on 11.05.2016.
 */
public class Layer {
    private List<Neuron> neurons;
    private List<Double> output;

    private List<Double> sigmas;

    private boolean isOutputLayer;

    public Layer(int neuronsNumber, int connectionsPerNeuron) {
        if (neuronsNumber <= 0)
            throw new IllegalArgumentException("Layer: should have 1 or more neurons");

        neurons = new ArrayList<>(neuronsNumber);
        output = new ArrayList<>();

        sigmas = new ArrayList<>();

        for (int i = 0; i < neuronsNumber; ++i) {
            neurons.add(new Neuron(connectionsPerNeuron));
        }

        isOutputLayer = false;
    }

    public Layer(List<Neuron> neurons) {
        if (neurons.isEmpty()) {
            throw new IllegalArgumentException("Layer: layer can't have zero neurons");
        }

        this.neurons = neurons;
        this.output = new ArrayList<>();
        this.sigmas = new ArrayList<>();

        isOutputLayer = false;
    }

    public List<Double> feedForward(List<Double> inputs) {
        if (inputs == null)
            throw new NullPointerException("Layer: inputs can't be null");

        output.clear();

        for (Neuron neuron : neurons) {
            double neuronOutput = neuron.feedForward(inputs);
            output.add(neuronOutput);
        }

        return output;
    }

    public void setAsOutputLayer() {
        this.isOutputLayer = true;
    }

    public void setAsHiddenLayer() { this.isOutputLayer = false; }

    public List<Double> getOutput() {
        return this.output;
    }

    public int size() {
        return this.neurons.size();
    }

    public Integer inputSize() {
        return neurons.get(0).getWeights().size();
    }

    public void backPropagateOutputLayer(List<Double> networkAnswer, List<Double> correctAnswer, double alpha) {
        if (!isOutputLayer) {
            throw new IllegalAccessError("Layer: layer is not an output layer");
        }

        if (networkAnswer.size() != neurons.size()) {
            throw new IllegalArgumentException("Layer: networkAnswer should have the same size with layer");
        }

        for (int i = 0; i < neurons.size(); ++i) {
            neurons.get(i).backPropagate(networkAnswer.get(i), correctAnswer.get(i), alpha);
        }

        sigmas = new ArrayList<>(neurons.size());

        for (Neuron neuron : neurons) {
            sigmas.add(neuron.getSigma());
        }
    }

    public void backPropagate(List<List<Double>> nextLayerWeights, List<Double> nextLayerSigmas, double alpha) {
        if (nextLayerWeights == null) {
            throw new NullPointerException("Layer: nextLayerWeights can't be null");
        }

        if (nextLayerSigmas == null) {
            throw new NullPointerException("Layer: nextLayerDeltas can't be null");
        }

        List<List<Double>> neuronsOutgoingWeights = getNeuronsOutgoingWeights(nextLayerWeights);

        for (int i = 0; i < neurons.size(); ++i) {
            neurons.get(i).backPropagate(neuronsOutgoingWeights.get(i), nextLayerSigmas, alpha);
        }

        sigmas = new ArrayList<>(neurons.size());

        for (Neuron neuron : neurons) {
            sigmas.add(neuron.getSigma());
        }
    }

    public void updateWeights() {
        for (Neuron neuron : neurons) {
            neuron.updateWeights();
        }
    }

    public List<List<Double>> getWeights() {
        List<List<Double>> weights = new ArrayList<>();

        for (Neuron neuron : neurons) {
            weights.add(neuron.getWeights());
        }

        return weights;
    }

    public void setWeights(List<List<Double>> weights) {
        if (weights == null)
            throw new NullPointerException("Layer: weights can't be null");

        if (neurons.size() != weights.size())
            throw new IllegalArgumentException("Layer: size of weights should be equal to number of neurons");

        for (int i = 0; i < neurons.size(); ++i) {
            neurons.get(i).setWeights(weights.get(i));
        }
    }

    public void setBias(double bias) {
        for (Neuron neuron : neurons) {
            neuron.setBias(bias);
        }
    }

    public List<Double> getSigmas() {
        return this.sigmas;
    }

    public static Layer parseLayer(String s) {
        String[] sNeurons = s.split("\n");

        if (sNeurons.length == 0) {
            throw new IllegalArgumentException("Layer: layer can't have zero weights");
        }

        List<Neuron> neurons = new ArrayList<>(sNeurons.length);

        for (String sNeuron : sNeurons) {
            Neuron neuron = Neuron.parseNeuron(sNeuron.trim());
            neurons.add(neuron);
        }

        return new Layer(neurons);
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder();

        for (int i = 0; i < neurons.size(); ++i) {
            text.append(neurons.get(i).toString());

            if (i < neurons.size() - 1)
                text.append("\n");
        }

        return text.toString();
    }

    private List<List<Double>> getNeuronsOutgoingWeights(List<List<Double>> nextLayerWeights) {
        List<List<Double>> neuronsOutgoingWeights = new ArrayList<>();

        for (int i = 0; i < neurons.size(); ++i) {
            List<Double> neuronOutgoingWeights = new ArrayList<>(nextLayerWeights.size());

            for (List<Double> weights : nextLayerWeights) {
                neuronOutgoingWeights.add(weights.get(i));
            }

            neuronsOutgoingWeights.add(neuronOutgoingWeights);
        }

        return neuronsOutgoingWeights;
    }
}
