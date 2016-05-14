package neuralnetwork;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Random;

/**
 * Created by Artem on 11.05.2016.
 */
public class Neuron {
    private List<Double> weights;
    private Double bias;
    private double output;
    private List<Double> input;

    private double sigma;
    private List<Double> deltas;

    private static final double RANGE_MAX = 0.5;
    private static final double RANGE_MIN = -0.5;

    public Neuron(int numberOfWeights) {
        if (numberOfWeights <= 0)
            throw new IllegalArgumentException("Neuron: should have 1 or more connections");

        weights = randomWeights(numberOfWeights);
        bias = randomDoubleInRange(RANGE_MIN, RANGE_MAX);
        output = -1;
        input = new ArrayList<>();

        sigma = 0;
        deltas = new ArrayList<>();

    }

    public Neuron(List<Double> weights) {
        this.weights = weights;
        this.bias = randomDoubleInRange(RANGE_MIN, RANGE_MAX);
        this.output = -1;

        sigma = 0;
        deltas = new ArrayList<>();
    }

    public Neuron(List<Double> weights, double bias) {
        this.weights = weights;
        this.bias = bias;
        this.output = -1;

        sigma = 0;
        deltas = new ArrayList<>();
    }

    public List<Double> getWeights() {
        return this.weights;
    }

    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    public double feedForward(List<Double> inputs) {
        if (inputs == null)
            throw new NullPointerException("Neuron: inputs can't be null");

        this.input = inputs;

        if (inputs.size() != weights.size())
            throw new IllegalArgumentException("Neuron: inputs and weights should be of the same size");

        double sum = sum(inputs);
        return sigmoid(sum);
    }

    public void backPropagate(List<Double> outgoingWeights, List<Double> nextLayerSigmas, double alpha) {
        if (outgoingWeights == null)
            throw new NullPointerException("Neuron: outgoingWeights can't be null");

        if (nextLayerSigmas == null)
            throw new NullPointerException("Neuron: nextLayerSigmas can't be null");

        if (outgoingWeights.size() != nextLayerSigmas.size()) {
            throw new IllegalArgumentException("Neuron: outgoingWeights and nextLayerSigmas should be of the same size");
        }

        double totalError = 0;

        for (int i = 0; i < outgoingWeights.size(); ++i) {
            totalError += outgoingWeights.get(i) * nextLayerSigmas.get(i);
        }

        this.sigma = totalError * derivative();

        deltas = new ArrayList<>();

        for (Double anInput : input) {
            double delta = alpha * this.sigma * anInput;
            deltas.add(delta);
        }

        bias = alpha * this.sigma;
    }

    public void backPropagate(Double neuronAnswer, Double correctAnswer, double alpha) {
        double error = correctAnswer - neuronAnswer;

        this.sigma = error * derivative();

        deltas = new ArrayList<>();

        for (Double anInput : input) {
            double delta = alpha * this.sigma * anInput;
            deltas.add(delta);
        }

        bias = alpha * this.sigma;
    }

    public void updateWeights() {
        for (int i = 0; i < weights.size(); ++i) {
            double oldWeight = weights.get(i);
            double newWeight = oldWeight + deltas.get(i);

            weights.set(i, newWeight);
        }
    }

    public double getSigma() {
        return sigma;
    }

    public double getOutput() {
        return this.output;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public static Neuron parseNeuron(String s) {
        String[] sWeights = s.split(" ");

        if (sWeights.length == 0) {
            throw new IllegalArgumentException("Neuron: neuron can't have zero weights");
        }

        List<Double> weights = new ArrayList<>(sWeights.length - 1);

        for (int i = 0; i < sWeights.length - 1; ++i) {
            Double weight = Double.parseDouble(sWeights[i].trim());
            weights.add(weight);
        }

        double bias = Double.parseDouble(sWeights[sWeights.length - 1]);

        return new Neuron(weights, bias);
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder();

        for (Double weight : weights) {
            text.append(weight + " ");
        }

        text.append(bias);

        return text.toString();
    }

    private double sum(List<Double> inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.size(); ++i) {
            sum += inputs.get(i) * weights.get(i);
        }

        sum += bias;

        return sum;
    }

    private double sigmoid(double sum) {
        this.output =  1.0 / (1.0 + Math.exp(-sum));
        return output;
    }

    private double derivative() {
        return this.output * (1.0 - this.output);
    }

    private List<Double> randomWeights(int numberOfWeights) {
        List<Double> weights = new ArrayList<>(numberOfWeights);

        for (int i = 0; i < numberOfWeights; ++i) {
            double randomWeight = randomDoubleInRange(RANGE_MIN, RANGE_MAX);
            weights.add(randomWeight);
        }

        return weights;
    }

    private double randomDoubleInRange(double rangeMin, double rangeMax) {
        Random random = new Random();
        return rangeMin + (rangeMax - rangeMin) * random.nextDouble();
    }
}
