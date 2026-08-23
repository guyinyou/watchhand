import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * WatchHand Java Server - TCP receiver with Swing visualization.
 * 实时预测依赖 ONNX Runtime（server/lib/onnxruntime-*.jar），
 * 模型由 server/train/export_onnx.py 从 last.pt 导出。
 *
 * Usage:
 *   javac -cp "lib/*" WatchHandServer.java
 *   java -cp ".:lib/*" WatchHandServer [port] [model.onnx]
 */
public class WatchHandServer {

    // Signal processing parameters
    static final double F_MIN = 18000.0;
    static final double F_MAX = 20000.0;
    static final int DISTANCE_BINS = 60;
    static final int TIME_WINDOW_FRAMES = 96;

    /** Chirp 长度：588 samples = 13.333ms @ 统一 44.1kHz，帧率 75fps（与客户端 AudioManager.CHIRP_LENGTH 保持同步） */
    static final int CHIRP_LENGTH = 588;

    /** 直达声锚定 bin，与 train/extract.py 的 DIRECT_BIN 保持一致 */
    static final int DIRECT_BIN = 5;

    // Shared state between threads
    static float[][] originalProfile = null;  // [distBins][frames]
    static float[][] diffProfile = null;      // [distBins][frames-1]
    static int frameCount = 0;
    static final Object dataLock = new Object();

    // Data collection state: raw PCM appended as received, labels stored as change events
    static int currentLabel = 0;
    static volatile boolean isCollecting = false;
    static OutputStream rawOut = null;
    static String sessionBaseName = null;
    static long collectionStartSamples = 0;
    static long collectionStartMs = 0;
    static final java.util.List<long[]> labelEvents = new java.util.ArrayList<>(); // {sampleOffsetInFile, label}
    static final java.util.concurrent.atomic.AtomicLong totalSamplesReceived = new java.util.concurrent.atomic.AtomicLong();
    static final java.util.concurrent.atomic.AtomicLong collectedBytes = new java.util.concurrent.atomic.AtomicLong();
    static int sampleRateForSave = 44100;
    static JLabel labelDisplay;
    static JLabel predDisplay;
    static JCheckBox testSetBox;
    static File sessionDir;

    // 实时预测：ONNX Runtime 加载 train/last.onnx（密集时间头，末步 argmax）
    static OrtEnvironment ortEnv = null;
    static OrtSession ortSession = null;
    static int predNumClasses = 0;
    static long predRuns = 0;

    // Colormap: blue -> cyan -> magenta -> yellow (matches paper)
    static int[] colorMapR = new int[256];
    static int[] colorMapG = new int[256];
    static int[] colorMapB = new int[256];

    static {
        for (int i = 0; i < 256; i++) {
            float t = i / 255f;
            if (t < 0.333f) {
                float s = t / 0.333f;
                colorMapR[i] = (int)(0 * 255);
                colorMapG[i] = (int)(0.7f * s * 255);
                colorMapB[i] = (int)((0.8f + 0.2f * s) * 255);
            } else if (t < 0.667f) {
                float s = (t - 0.333f) / 0.334f;
                colorMapR[i] = (int)(0.7f * s * 255);
                colorMapG[i] = (int)(0.7f * (1f - s) * 255);
                colorMapB[i] = (int)((1f - 0.2f * s) * 255);
            } else {
                float s = (t - 0.667f) / 0.333f;
                colorMapR[i] = (int)((0.7f + 0.3f * s) * 255);
                colorMapG[i] = (int)(0.7f * s * 255);
                colorMapB[i] = (int)(0.8f * (1f - s) * 255);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        int port = args.length > 0 ? Integer.parseInt(args[0]) : 9999;
        String modelPath = args.length > 1 ? args[1] : "train/last.onnx";
        loadPredictor(modelPath);

        System.out.println("WatchHand Java Server");
        System.out.println("Listening on port " + port + "...");

        // Build UI on EDT
        SwingUtilities.invokeLater(() -> buildUI());

        // TCP server on main thread - loops to accept reconnections
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            while (true) {
                System.out.println("\nWaiting for client connection...");
                Socket client = serverSocket.accept();
                System.out.println("Connected: " + client.getInetAddress());

                try {
                    DataInputStream in = new DataInputStream(client.getInputStream());

                    // Read header
                    StringBuilder header = new StringBuilder();
                    int b;
                    while ((b = in.read()) != '\n') {
                        if (b == -1) break;
                        header.append((char) b);
                    }
                    System.out.println("Header: " + header.toString().trim());

                    // Read sample rate (4 bytes LE)
                    byte[] srBytes = new byte[4];
                    in.readFully(srBytes);
                    int sampleRate = ByteBuffer.wrap(srBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    sampleRateForSave = sampleRate;
                    System.out.println("Sample rate: " + sampleRate + " Hz");

                    // Read config (channels, bits)
                    int channels = in.read();
                    int bitsPerSample = in.read();
                    System.out.println("Channels: " + channels + ", Bits: " + bitsPerSample);

                    // Initialize processor (统一 44.1kHz，chirp 588 samples，与客户端一致)
                    EchoProfileProcessor processor = new EchoProfileProcessor(
                        sampleRate, F_MIN, F_MAX, CHIRP_LENGTH, DISTANCE_BINS, TIME_WINDOW_FRAMES
                    );
                    System.out.println("Processor initialized (chirp=" + CHIRP_LENGTH + " samples). Waiting for data...");

                    // Read PCM data and process (50ms read buffer, 按连接采样率动态计算)
                    byte[] readBuffer = new byte[sampleRate / 10];
                    totalSamplesReceived.set(0);
                    long startTime = System.currentTimeMillis();
                    int framesProduced = 0;
                    long lastByteTime = System.currentTimeMillis();

                    while (true) {
                        int bytesRead = in.read(readBuffer);
                        if (bytesRead <= 0) break;

                        // 流间断 > 500ms = 客户端音频会话重启（stop->start），
                        // 重建处理器以对齐新会话的 tx/rx 相位（与本地生命周期一致）
                        long now = System.currentTimeMillis();
                        if (now - lastByteTime > 500) {
                            processor = new EchoProfileProcessor(
                                sampleRate, F_MIN, F_MAX, CHIRP_LENGTH, DISTANCE_BINS, TIME_WINDOW_FRAMES);
                            System.out.println("Stream gap " + (now - lastByteTime) + "ms: processor realigned to new audio session");
                        }
                        lastByteTime = now;

                        int numSamples = bytesRead / 2;
                        short[] samples = new short[numSamples];
                        ByteBuffer.wrap(readBuffer, 0, bytesRead)
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .asShortBuffer()
                            .get(samples);

                        totalSamplesReceived.addAndGet(numSamples);

                        // Collection = append raw bytes as received (no windowing involved)
                        if (isCollecting && rawOut != null) {
                            rawOut.write(readBuffer, 0, bytesRead);
                            collectedBytes.addAndGet(bytesRead);
                        }

                        float[][] result = processor.feed(samples);
                        if (result != null) {
                            float[][] origCopy;
                            synchronized (dataLock) {
                                originalProfile = result[0] != null ? reshapeTo2D(result[0], DISTANCE_BINS, TIME_WINDOW_FRAMES) : null;
                                diffProfile = result[1] != null ? reshapeTo2D(result[1], DISTANCE_BINS, TIME_WINDOW_FRAMES - 1) : null;
                                frameCount = TIME_WINDOW_FRAMES;
                                origCopy = originalProfile;
                            }

                            // 实时预测：每 2 帧跑一次 ONNX，结果显示到 Pred label
                            if (ortSession != null && origCopy != null && framesProduced % 2 == 0) {
                                final int pred = predictClass(origCopy);
                                if (pred >= 0) {
                                    SwingUtilities.invokeLater(() -> {
                                        predDisplay.setText("Pred: " + pred);
                                        predDisplay.setBackground(labelColor(pred));
                                    });
                                }
                            }

                            framesProduced++;
                            if (framesProduced % 5 == 0) {
                                double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                                long totalSamples = totalSamplesReceived.get();
                                System.out.printf("[%.1fs] Frames: %d, Samples: %d, Rate: %.1fk samples/s%n",
                                    elapsed, framesProduced, totalSamples, totalSamples / elapsed / 1000.0);
                            }
                        }
                    }

                    System.out.println("Client disconnected. Total frames: " + framesProduced);
                    if (isCollecting) stopCollection();  // auto-save on disconnect
                } catch (Exception e) {
                    System.out.println("Connection error: " + e.getMessage());
                } finally {
                    client.close();
                }
                // Loop back to accept next connection
            }
        }
    }

    static float[][] reshapeTo2D(float[] flat, int rows, int cols) {
        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[r][c] = flat[r * cols + c];
            }
        }
        return result;
    }

    static synchronized void startCollection() throws IOException {
        if (isCollecting) return;

        sessionDir = new File("collected_data", testSetBox != null && testSetBox.isSelected() ? "test" : "train");
        if (!sessionDir.exists()) sessionDir.mkdirs();

        String timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date());
        sessionBaseName = "gesture_data_" + timestamp;
        rawOut = new BufferedOutputStream(new FileOutputStream(new File(sessionDir, sessionBaseName + ".raw")));

        synchronized (labelEvents) {
            labelEvents.clear();
            labelEvents.add(new long[]{0, currentLabel});  // initial label at file offset 0
        }
        collectedBytes.set(0);
        collectionStartSamples = totalSamplesReceived.get();
        collectionStartMs = System.currentTimeMillis();
        isCollecting = true;
        System.out.println("Collection started [" + sessionDir.getName() + "] -> " + sessionDir.getPath() + "/" + sessionBaseName + ".raw");
    }

    static synchronized void stopCollection() {
        if (!isCollecting) return;
        isCollecting = false;

        try {
            if (rawOut != null) { rawOut.flush(); rawOut.close(); rawOut = null; }
        } catch (IOException e) {
            System.out.println("Error closing raw file: " + e.getMessage());
        }

        long bytes = collectedBytes.get();
        if (bytes == 0) {
            System.out.println("No data collected, nothing saved.");
            return;
        }

        File dataDir = sessionDir;
        try (PrintWriter labelOut = new PrintWriter(new FileWriter(new File(dataDir, sessionBaseName + ".labels")));
             PrintWriter metaOut = new PrintWriter(new FileWriter(new File(dataDir, sessionBaseName + ".meta")))) {

            // Label change events: "sampleOffsetInFile label"
            synchronized (labelEvents) {
                for (long[] ev : labelEvents) labelOut.println(ev[0] + " " + ev[1]);
            }

            long numSamples = bytes / 2;
            metaOut.println("sample_rate=" + sampleRateForSave);
            metaOut.println("channels=1");
            metaOut.println("bits=16");
            metaOut.println("num_raw_samples=" + numSamples);
            metaOut.println("duration_s=" + String.format("%.2f", numSamples / (double) sampleRateForSave));
            metaOut.println("f_min=" + (int) F_MIN);
            metaOut.println("f_max=" + (int) F_MAX);
            metaOut.println("chirp_length=" + CHIRP_LENGTH);
            metaOut.println("chirp_duration_ms=" + String.format(java.util.Locale.US, "%.3f", CHIRP_LENGTH * 1000.0 / sampleRateForSave));
            metaOut.println("distance_bins=" + DISTANCE_BINS);
            metaOut.println("time_window_frames=" + TIME_WINDOW_FRAMES);
            metaOut.println("started_at=" + new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date(collectionStartMs)));

            System.out.println("Saved " + numSamples + " raw samples (" + String.format("%.1f", bytes / 1e6) + " MB)");
            System.out.println("  Raw:    " + sessionBaseName + ".raw");
            System.out.println("  Labels: " + sessionBaseName + ".labels");
            System.out.println("  Meta:   " + sessionBaseName + ".meta");
        } catch (Exception e) {
            System.out.println("Error saving data: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /** Label display color: 0 gray, 1 green, 2 red, others blue. */
    static Color labelColor(int lab) {
        switch (lab) {
            case 0: return Color.DARK_GRAY;
            case 1: return new Color(0, 100, 0);
            case 2: return new Color(140, 0, 0);
            default: return new Color(0, 0, 120);
        }
    }

    /** Change current label; records a change event while collecting. EDT-only. */
    static void setLabel(int newLabel) {
        if (newLabel == currentLabel) return;
        currentLabel = newLabel;
        if (isCollecting) {
            synchronized (labelEvents) {
                labelEvents.add(new long[]{totalSamplesReceived.get() - collectionStartSamples, newLabel});
            }
        }
        if (labelDisplay != null) {
            labelDisplay.setText("Label: " + newLabel);
            labelDisplay.setBackground(labelColor(newLabel));
        }
    }

    /** 加载 ONNX 实时预测模型；文件不存在则禁用预测。 */
    static void loadPredictor(String path) {
        File f = new File(path);
        if (!f.exists()) {
            System.out.println("Predictor disabled: " + f.getAbsolutePath() + " not found"
                + " (run server/train/export_onnx.py first)");
            return;
        }
        try {
            ortEnv = OrtEnvironment.getEnvironment();
            ortSession = ortEnv.createSession(path, new OrtSession.SessionOptions());
            System.out.println("Predictor loaded: " + path);
        } catch (Exception e) {
            System.out.println("Predictor load failed: " + e.getMessage());
            ortSession = null;
        }
    }

    /**
     * 实时预测：用当前滚动窗口构造双通道输入跑 ONNX，返回末步 argmax 类别，失败 -1。
     * 与 train.py 推理完全一致：距离轴 edge pad 60->64、差分通道、逐通道窗口内归一化。
     */
    static int predictClass(float[][] orig) {
        float[] last = lastStepLogits(orig);
        if (last == null) return -1;
        int best = 0;
        for (int k = 1; k < last.length; k++) if (last[k] > last[best]) best = k;
        predRuns++;
        return best;
    }

    /** 构造与 train.py 推理一致的双通道归一化输入 [2][64][96]。 */
    static float[][][] buildInput(float[][] orig) {
        int bins = DISTANCE_BINS, T = TIME_WINDOW_FRAMES, pad = 64;
        float[][][] x = new float[2][pad][T];
        // 通道 0：原始轮廓（edge pad）
        for (int d = 0; d < pad; d++) {
            int sd = Math.min(d, bins - 1);
            for (int t = 0; t < T; t++) x[0][d][t] = orig[sd][t];
        }
        // 通道 1：差分 |P[f+1]|-|P[f]|，末列重复
        for (int d = 0; d < pad; d++) {
            int sd = Math.min(d, bins - 1);
            for (int t = 0; t < T - 1; t++)
                x[1][d][t] = Math.abs(orig[sd][t + 1]) - Math.abs(orig[sd][t]);
            x[1][d][T - 1] = x[1][d][T - 2];
        }
        // 逐通道窗口内归一化
        for (int c = 0; c < 2; c++) {
            double mu = 0, sq = 0;
            for (int d = 0; d < pad; d++) for (int t = 0; t < T; t++) mu += x[c][d][t];
            mu /= pad * T;
            for (int d = 0; d < pad; d++) for (int t = 0; t < T; t++) {
                double v = x[c][d][t] - mu; sq += v * v;
            }
            double sdv = Math.sqrt(sq / (pad * T)) + 1e-6;
            for (int d = 0; d < pad; d++) for (int t = 0; t < T; t++)
                x[c][d][t] = (float) ((x[c][d][t] - mu) / sdv);
        }
        return x;
    }

    /** 末步 logits；失败 null。 */
    static float[] lastStepLogits(float[][] orig) {
        if (ortSession == null || orig == null) return null;
        try {
            float[][][] x = buildInput(orig);
            try (OnnxTensor in = OnnxTensor.createTensor(ortEnv, new float[][][][]{x});
                 OrtSession.Result res = ortSession.run(java.util.Collections.singletonMap("x", in))) {
                float[][][] logits = (float[][][]) res.get(0).getValue();
                predNumClasses = logits[0].length;
                return logits[0][logits[0].length - 1];
            }
        } catch (Exception e) {
            return null;
        }
    }

    /** Protocol-driven label event (guided collection): records even if same as current. */
    static void recordLabelEvent(int lab) {
        currentLabel = lab;
        if (isCollecting) {
            synchronized (labelEvents) {
                labelEvents.add(new long[]{totalSamplesReceived.get() - collectionStartSamples, lab});
            }
        }
        if (labelDisplay != null) {
            labelDisplay.setText("Label: " + lab);
            labelDisplay.setBackground(labelColor(lab));
        }
    }

    static double parseD(String s, double def) {
        try { return Double.parseDouble(s.trim()); } catch (Exception e) { return def; }
    }

    static void buildUI() {
        JFrame frame = new JFrame("WatchHand - Echo Profile Visualization");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Main panel with BorderLayout
        JPanel mainPanel = new JPanel(new BorderLayout());

        // Top panel for label display and controls
        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 10));

        // Label display
        labelDisplay = new JLabel("Label: 0", SwingConstants.CENTER);
        labelDisplay.setFont(new Font("SansSerif", Font.BOLD, 40));
        labelDisplay.setForeground(Color.WHITE);
        labelDisplay.setBackground(Color.DARK_GRAY);
        labelDisplay.setOpaque(true);
        labelDisplay.setPreferredSize(new Dimension(620, 80));
        topPanel.add(labelDisplay);

        // 实时预测显示
        predDisplay = new JLabel("Pred: -", SwingConstants.CENTER);
        predDisplay.setFont(new Font("SansSerif", Font.BOLD, 40));
        predDisplay.setForeground(Color.WHITE);
        predDisplay.setBackground(Color.DARK_GRAY);
        predDisplay.setOpaque(true);
        predDisplay.setPreferredSize(new Dimension(220, 80));
        topPanel.add(predDisplay);

        // Collection toggle button
        JButton collectBtn = new JButton("Start Collection");
        collectBtn.setFont(new Font("SansSerif", Font.BOLD, 20));
        collectBtn.setPreferredSize(new Dimension(200, 60));
        topPanel.add(collectBtn);

        // Sample count display
        JLabel sampleCountLabel = new JLabel("Samples: 0", SwingConstants.CENTER);
        sampleCountLabel.setFont(new Font("SansSerif", Font.PLAIN, 16));
        topPanel.add(sampleCountLabel);

        mainPanel.add(topPanel, BorderLayout.NORTH);

        // Guided (protocol-driven) cyclic collection, learned from wlemg cycle recording:
        // labels come from the protocol timer, not manual key presses.
        JPanel guidedPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 8, 5));
        JTextField classesField = new JTextField("1,2", 8);
        JTextField trialSecField = new JTextField("2.5", 4);
        JTextField roundsField = new JTextField("5", 3);
        testSetBox = new JCheckBox("test set", false);
        JButton guidedBtn = new JButton("Start Guided Collection");
        guidedBtn.setFont(new Font("SansSerif", Font.BOLD, 14));
        guidedPanel.add(new JLabel("classes")); guidedPanel.add(classesField);
        guidedPanel.add(new JLabel("trial(s)")); guidedPanel.add(trialSecField);
        guidedPanel.add(new JLabel("rounds")); guidedPanel.add(roundsField);
        guidedPanel.add(testSetBox);
        guidedPanel.add(guidedBtn);
        mainPanel.add(guidedPanel, BorderLayout.SOUTH);

        // ---- guided state machine (EDT timers)
        final java.util.List<Integer>[] seqHolder = new java.util.List[1];
        final int[] idx = {0};
        final boolean[] running = {false};
        final javax.swing.Timer[] timers = new javax.swing.Timer[2];  // [phase, tick]

        final Runnable finishGuided = () -> {
            running[0] = false;
            if (timers[0] != null) timers[0].stop();
            if (timers[1] != null) timers[1].stop();
            stopCollection();
            guidedBtn.setText("Start Guided Collection");
            guidedBtn.setBackground(null);
            classesField.setEditable(true); trialSecField.setEditable(true);
            roundsField.setEditable(true);
            collectBtn.setEnabled(true);
            labelDisplay.setText("Label: 0");
            labelDisplay.setBackground(Color.DARK_GRAY);
        };

        final Runnable[] nextTrial = new Runnable[1];
        nextTrial[0] = () -> {
            if (!running[0]) return;
            if (idx[0] >= seqHolder[0].size()) { finishGuided.run(); return; }
            int lab = seqHolder[0].get(idx[0]);
            double trialSec = parseD(trialSecField.getText(), 2.5);
            java.awt.Toolkit.getDefaultToolkit().beep();
            recordLabelEvent(lab);
            timers[0] = new javax.swing.Timer((int) (trialSec * 1000), ev -> {
                idx[0]++;
                // 无 rest、边界不插 0：标签在 trial 切换点直接跳到下一类
                nextTrial[0].run();
            });
            timers[0].setRepeats(false);
            timers[0].start();
        };

        guidedBtn.addActionListener(e -> {
            if (running[0]) { finishGuided.run(); return; }
            java.util.List<Integer> classes = new java.util.ArrayList<>();
            for (String s : classesField.getText().split("[,，\\s]+")) {
                if (s.isEmpty()) continue;
                try {
                    classes.add(Integer.parseInt(s));
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(frame, "Invalid class list: " + classesField.getText());
                    return;
                }
            }
            if (classes.isEmpty()) { JOptionPane.showMessageDialog(frame, "Class list is empty"); return; }
            int rounds;
            try { rounds = Math.max(1, Integer.parseInt(roundsField.getText().trim())); }
            catch (NumberFormatException ex) { rounds = 5; }
            java.util.List<Integer> seq = new java.util.ArrayList<>();
            for (int r = 0; r < rounds; r++) seq.addAll(classes);
            seqHolder[0] = seq;
            idx[0] = 0;
            try {
                startCollection();
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(frame, "Failed to start collection: " + ex.getMessage());
                return;
            }
            running[0] = true;
            guidedBtn.setText("Stop & Save");
            guidedBtn.setBackground(Color.RED);
            classesField.setEditable(false); trialSecField.setEditable(false);
            roundsField.setEditable(false);
            collectBtn.setEnabled(false);
            recordLabelEvent(0);
            timers[0] = new javax.swing.Timer(2000, ev -> nextTrial[0].run());
            timers[0].setRepeats(false);
            timers[0].start();
        });

        // Heatmap panel
        JPanel heatmapContainer = new JPanel(new GridLayout(2, 1));
        HeatmapPanel origPanel = new HeatmapPanel("Original Echo Profile");
        HeatmapPanel diffPanel = new HeatmapPanel("Differential Echo Profile");
        heatmapContainer.add(origPanel);
        heatmapContainer.add(diffPanel);
        mainPanel.add(heatmapContainer, BorderLayout.CENTER);

        frame.setContentPane(mainPanel);
        frame.setSize(1200, 850);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.requestFocusInWindow();

        // Global key listener: hold digit 1-9 (or a=1/s=2) to label, release -> 0
        java.awt.KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(e -> {
            if (e.getID() == java.awt.event.KeyEvent.KEY_PRESSED) {
                char key = e.getKeyChar();
                if (key >= '1' && key <= '9') setLabel(key - '0');
                else if (key == 'a' || key == 'A') setLabel(1);
                else if (key == 's' || key == 'S') setLabel(2);
            } else if (e.getID() == java.awt.event.KeyEvent.KEY_RELEASED) {
                char key = Character.toLowerCase(e.getKeyChar());
                if ((key >= '1' && key <= '9') || key == 'a' || key == 's') setLabel(0);
            }
            return false;
        });

        // Collection button action
        collectBtn.addActionListener(e -> {
            if (!isCollecting) {
                try {
                    startCollection();
                    collectBtn.setText("Stop & Save");
                    collectBtn.setBackground(Color.RED);
                } catch (IOException ex) {
                    System.out.println("Failed to start collection: " + ex.getMessage());
                }
            } else {
                stopCollection();
                collectBtn.setText("Start Collection");
                collectBtn.setBackground(null);
            }
        });

        // Update loop at 10 FPS
        Timer timer = new Timer(100, e -> {
            synchronized (dataLock) {
                origPanel.setData(originalProfile);
                diffPanel.setData(diffProfile);
            }
            origPanel.repaint();
            diffPanel.repaint();
            long cb = collectedBytes.get();
            sampleCountLabel.setText(String.format("Collected: %.1f MB (%.0fs)", cb / 1e6, cb / 2.0 / sampleRateForSave));
        });
        timer.start();
    }

    static class HeatmapPanel extends JPanel {
        String title;
        float[][] data;

        HeatmapPanel(String title) {
            this.title = title;
            setBackground(Color.BLACK);
        }

        void setData(float[][] data) {
            this.data = data;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            int w = getWidth();
            int h = getHeight();

            // Draw title
            g2.setColor(Color.WHITE);
            g2.setFont(new Font("SansSerif", Font.BOLD, 14));
            g2.drawString(title, 10, 20);

            if (data == null) {
                g2.setColor(Color.GRAY);
                g2.setFont(new Font("SansSerif", Font.PLAIN, 16));
                FontMetrics fm = g2.getFontMetrics();
                String msg = "Waiting for data...";
                g2.drawString(msg, (w - fm.stringWidth(msg)) / 2, h / 2);
                return;
            }

            int rows = data.length;
            int cols = data[0].length;

            // Percentile clipping
            float[] sorted = new float[rows * cols];
            int idx = 0;
            for (float[] row : data) for (float v : row) sorted[idx++] = v;
            java.util.Arrays.sort(sorted);
            float lo = sorted[(int)(sorted.length * 0.02)];
            float hi = sorted[(int)(sorted.length * 0.98)];
            float range = hi - lo;
            if (range < 1e-10f) range = 1f;

            // Draw title with range
            g2.setColor(Color.WHITE);
            g2.drawString(String.format("%s (Range: %.2e ~ %.2e)", title, lo, hi), 10, 20);

            // Draw heatmap
            int margin = 10;
            int titleH = 30;
            int drawW = w - 2 * margin;
            int drawH = h - titleH - 2 * margin;
            float cellW = (float) drawW / cols;
            float cellH = (float) drawH / rows;

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    float clipped = Math.max(lo, Math.min(hi, data[r][c]));
                    float normalized = (clipped - lo) / range;
                    int colorIdx = (int)(normalized * 255);
                    colorIdx = Math.max(0, Math.min(255, colorIdx));

                    g2.setColor(new Color(colorMapR[colorIdx], colorMapG[colorIdx], colorMapB[colorIdx]));
                    int x = margin + (int)(c * cellW);
                    int y = titleH + margin + (int)(r * cellH);
                    g2.fillRect(x, y, (int)(cellW + 1), (int)(cellH + 1));
                }
            }
        }
    }

    /**
     * Echo profile processor - matches paper Algorithm 1.
     */
    static class EchoProfileProcessor {
        int sampleRate;
        double fMin, fMax;
        int chirpLength, distanceBins, timeWindowFrames;
        float[] txChirp;

        // Biquad filter
        float[] hpCoeffs, lpCoeffs;
        float[] hpZ1 = new float[3], hpZ2 = new float[3];
        float[] lpZ1 = new float[2], lpZ2 = new float[2];

        // 不足一个 chirp 的零头先暂存，凑满整 chirp 才做分段相关（与 APK 端一致）
        float[] pendingBuf;
        int pendingPos;

        // Start alignment
        int startOffset = -1;
        float[] alignBuf;
        int alignPos;
        boolean startFound = false;
        int alignPass = 0;  // SNR 连续通过计数：满 8 个 chirp 边界才锁定，保证对齐缓冲全是信号

        // 逐 chirp 分段相关（与 APK 端优化一致：每帧 60 bins × 600 乘加 ≈ 0.05ms）
        float[] prevChunks;
        float[] segBuf;
        float[] rollingOrig;
        float[] rollingDiff;
        float[] newCol;
        int producedFrames;

        EchoProfileProcessor(int sampleRate, double fMin, double fMax,
                            int chirpLength, int distanceBins, int timeWindowFrames) {
            this.sampleRate = sampleRate;
            this.fMin = fMin;
            this.fMax = fMax;
            this.chirpLength = chirpLength;
            this.distanceBins = distanceBins;
            this.timeWindowFrames = timeWindowFrames;

            txChirp = generateChirp();
            hpCoeffs = computeHighpassCoeffs(fMin, sampleRate);
            lpCoeffs = computeLowpassCoeffs(fMax, sampleRate);

            pendingBuf = new float[chirpLength];
            prevChunks = new float[chirpLength * 2];
            segBuf = new float[chirpLength * 3];
            rollingOrig = new float[distanceBins * timeWindowFrames];
            rollingDiff = new float[distanceBins * (timeWindowFrames - 1)];
            newCol = new float[distanceBins];
            alignBuf = new float[chirpLength * 4];
        }

        float[] generateChirp() {
            float[] signal = new float[chirpLength];
            double T = (double) chirpLength / sampleRate;
            for (int n = 0; n < chirpLength; n++) {
                double t = (double) n / sampleRate;
                double phase = 2 * Math.PI * (fMin * t + (fMax - fMin) * t * t / (2 * T));
                // Hann window
                double window = 0.5 * (1 - Math.cos(2 * Math.PI * n / (chirpLength - 1)));
                signal[n] = (float)(Math.cos(phase) * window);
            }
            return signal;
        }

        float[] computeHighpassCoeffs(double fc, int fs) {
            double w0 = 2 * Math.PI * fc / fs;
            double cosW0 = Math.cos(w0), sinW0 = Math.sin(w0);
            double alpha = sinW0 / (2 * 0.707);
            double a0 = 1 + alpha;
            return new float[]{
                (float)((1 + cosW0) / 2 / a0),
                (float)(-(1 + cosW0) / a0),
                (float)((1 + cosW0) / 2 / a0),
                (float)(-2 * cosW0 / a0),
                (float)((1 - alpha) / a0)
            };
        }

        float[] computeLowpassCoeffs(double fc, int fs) {
            double w0 = 2 * Math.PI * fc / fs;
            double cosW0 = Math.cos(w0), sinW0 = Math.sin(w0);
            double alpha = sinW0 / (2 * 0.707);
            double a0 = 1 + alpha;
            return new float[]{
                (float)((1 - cosW0) / 2 / a0),
                (float)((1 - cosW0) / a0),
                (float)((1 - cosW0) / 2 / a0),
                (float)(-2 * cosW0 / a0),
                (float)((1 - alpha) / a0)
            };
        }

        float biquadStep(float x, float[] coeffs, float[] z1arr, float[] z2arr, int idx) {
            float b0 = coeffs[0], b1 = coeffs[1], b2 = coeffs[2];
            float a1 = coeffs[3], a2 = coeffs[4];
            float y = b0 * x + z1arr[idx];
            z1arr[idx] = b1 * x - a1 * y + z2arr[idx];
            z2arr[idx] = b2 * x - a2 * y;
            return y;
        }

        /**
         * Feed samples. Returns [originalProfile, diffProfile] or null.
         * 与 APK 端一致：逐 chirp 分段相关 + 滚动 96 帧窗口。
         */
        float[][] feed(short[] samples) {
            // Filter and accumulate
            for (short s : samples) {
                float x = s;
                for (int i = 0; i < 3; i++) x = biquadStep(x, hpCoeffs, hpZ1, hpZ2, i);
                for (int i = 0; i < 2; i++) x = biquadStep(x, lpCoeffs, lpZ1, lpZ2, i);

                if (!startFound) {
                    alignBuf[alignPos % alignBuf.length] = x;
                    alignPos++;
                    // 缓冲满后按 chirp 边界重试对齐：信号未到（SNR 不足）就继续等，
                    // 避免“先连接后开音”时锁定在纯噪声上
                    if (alignPos >= alignBuf.length && alignPos % chirpLength == 0) {
                        int off = findStartOffset();
                        if (off >= 0) {
                            // 信号刚到时缓冲只有一部分是信号，需连续 8 次通过才锁定
                            if (++alignPass >= 8) {
                                startOffset = off;
                                startFound = true;
                            }
                        } else {
                            alignPass = 0;
                        }
                    }
                    continue;
                }

                pendingBuf[pendingPos++] = x;
                if (pendingPos < chirpLength) continue;  // 零头先留着，凑满一个 chirp 再做分段相关
                processChunk();
                pendingPos = 0;
            }

            if (producedFrames < timeWindowFrames) return null;
            return new float[][]{rollingOrig.clone(), rollingDiff.clone()};
        }

        /** 每凑满一个 chirp 产出一帧（与 APK 端 processChunk 一致） */
        private void processChunk() {
            int L = chirpLength;
            // seg = 前两个 chirp + 当前 chirp
            for (int i = 0; i < 2 * L; i++) segBuf[i] = prevChunks[i];
            for (int i = 0; i < L; i++) segBuf[2 * L + i] = pendingBuf[i];
            // 滚动 overlap 窗口
            for (int i = 0; i < L; i++) prevChunks[i] = prevChunks[i + L];
            for (int i = 0; i < L; i++) prevChunks[L + i] = pendingBuf[i];

            // 取 bin：相关约定 lag = startOffset + d（与 extract.py 互相关一致）
            for (int d = 0; d < distanceBins; d++) {
                int k = startOffset + d;
                float sum = 0f;
                for (int j = 0; j < L; j++) sum += segBuf[k + j] * txChirp[j];
                newCol[d] = sum;
            }

            // 96 帧窗口左移一列，新列追加到尾部（原始轮廓保留带符号值，论文约定）
            for (int d = 0; d < distanceBins; d++) {
                int base = d * timeWindowFrames;
                for (int fi = 0; fi < timeWindowFrames - 1; fi++) rollingOrig[base + fi] = rollingOrig[base + fi + 1];
                rollingOrig[base + timeWindowFrames - 1] = newCol[d];
            }
            // 差分轮廓滚动：diff[f] = |P[f]| - |P[f-1]|
            for (int d = 0; d < distanceBins; d++) {
                int base = d * (timeWindowFrames - 1);
                for (int fi = 0; fi < timeWindowFrames - 2; fi++) rollingDiff[base + fi] = rollingDiff[base + fi + 1];
                float curr = Math.abs(rollingOrig[d * timeWindowFrames + timeWindowFrames - 1]);
                float prev = Math.abs(rollingOrig[d * timeWindowFrames + timeWindowFrames - 2]);
                rollingDiff[base + timeWindowFrames - 2] = curr - prev;
            }
            producedFrames++;
        }

        int findStartOffset() {
            int L = chirpLength;
            int searchLen = Math.min(alignBuf.length + L - 1, L * 3);
            float[] corr = new float[searchLen];
            float sumAbs = 0; float maxAbs = 0;
            for (int k = 0; k < searchLen; k++) {
                float sum = 0;
                int jMax = Math.min(L - 1, alignBuf.length - 1 - k);
                for (int j = 0; j <= jMax; j++) sum += alignBuf[k + j] * txChirp[j];
                corr[k] = sum;
                float a = Math.abs(sum);
                sumAbs += a;
                if (a > maxAbs) maxAbs = a;
            }
            // SNR 门槛：直达声峰值需显著高于相关底噪（实测信号 ≈14，静默 ≈5）
            if (sumAbs <= 0 || maxAbs < 9 * sumAbs / searchLen) return -1;
            // b0：逐 chirp 平均幅度最大的 bin（与 extract.py batch_profile 一致）
            int n0 = (searchLen / L) * L;
            int b0 = 0; float best = -1;
            for (int d = 0; d < L; d++) {
                float s = 0;
                for (int k = d; k < n0; k += L) s += Math.abs(corr[k]);
                if (s > best) { best = s; b0 = d; }
            }
            // p_start 候选 + 直达峰验证（与 extract.py 一致）
            int pStart = (b0 - DIRECT_BIN % L + L) % L;
            for (int ci = 0; ci < 2; ci++) {
                int cand = ci == 0 ? (b0 - DIRECT_BIN + L) % L : (DIRECT_BIN - b0 + L) % L;
                int nf = (searchLen - cand) / L - 2;
                if (nf <= 0) continue;
                int am = 0; float amv = -1;
                for (int d = 0; d < distanceBins; d++) {
                    float s = 0;
                    for (int f2 = 0; f2 < nf; f2++) s += Math.abs(corr[cand + f2 * L + d]);
                    if (s > amv) { amv = s; am = d; }
                }
                if (am - DIRECT_BIN >= 0 && am - DIRECT_BIN <= 2) { pStart = cand; break; }
            }
            // 帧相位对齐：bin d 取 lag = startOffset + d，需 startOffset ≡ pStart (mod L)，
            // 使流式轮廓与训练批处理轮廓逐帧一致
            return pStart % L;
        }

    }
}
