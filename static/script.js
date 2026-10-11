document.addEventListener("DOMContentLoaded", () => {

    // ==============================
    // ELEMENTS
    // ==============================

    const uploadBtn = document.getElementById("chooseImage");
    const fileInput = document.getElementById("fruitInput");
    const preview = document.getElementById("preview");

    const openCamera = document.getElementById("openCamera");
    const cameraArea = document.getElementById("cameraArea");
    const video = document.getElementById("cameraVideo");
    const canvas = document.getElementById("cameraCanvas");
    const capturePhoto = document.getElementById("capturePhoto");
    const closeCamera = document.getElementById("closeCamera");

    const analyzeButton = document.getElementById("analyzeButton");

    // Result elements
    const freshnessScore = document.getElementById("freshnessScore");
    const freshnessMeter = document.getElementById("freshnessMeter");
    const freshnessRing = document.getElementById("freshnessRing");
    const scoreText = document.getElementById("scoreText");

    const predictionValue = document.getElementById("predictionValue");
    const conditionValue = document.getElementById("conditionValue");
    const shelfLifeValue = document.getElementById("shelfLifeValue");
    const confidenceValue = document.getElementById("confidenceValue");

    const resultTitle = document.getElementById("resultTitle");
    const resultDescription = document.getElementById("resultDescription");

    const recommendation = document.getElementById("recommendation");
    const recommendationText =
        document.getElementById("recommendationText");

    const whyPrediction = document.getElementById("whyPrediction");


    // ==============================
    // VARIABLES
    // ==============================

    let selectedImage = null;
    let cameraStream = null;


    // ==============================
    // UPLOAD IMAGE
    // ==============================

    if (uploadBtn && fileInput) {

        uploadBtn.addEventListener("click", () => {
            fileInput.click();
        });


        fileInput.addEventListener("change", () => {

            const file = fileInput.files[0];

            if (!file) {
                return;
            }

            selectedImage = file;

            const imageURL = URL.createObjectURL(file);

            preview.innerHTML = `
                <img
                    class="preview-image"
                    src="${imageURL}"
                    alt="Selected fruit"
                >

                <p class="success-text">
                    ✓ Fruit image uploaded successfully
                </p>
            `;

            resetPrediction();

        });
    }


    // ==============================
    // OPEN CAMERA
    // ==============================

    if (openCamera) {

        openCamera.addEventListener("click", async () => {

            try {

                if (
                    !navigator.mediaDevices ||
                    !navigator.mediaDevices.getUserMedia
                ) {
                    throw new Error("Camera API unavailable");
                }

                if (cameraStream) {
                    stopCamera();
                }

                cameraStream =
                    await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: {
                                ideal: "environment"
                            }
                        },
                        audio: false
                    });

                video.srcObject = cameraStream;

                cameraArea.style.display = "block";

                openCamera.textContent = "📷 Camera Open";

            }

            catch (error) {

                console.error("Camera Error:", error);

                alert(
                    "Camera could not be opened. " +
                    "Please allow camera permission and try again."
                );

            }

        });

    }


    // ==============================
    // CAPTURE PHOTO
    // ==============================

    if (capturePhoto) {

        capturePhoto.addEventListener("click", () => {

            if (!cameraStream) {

                alert("Please open the camera first.");

                return;
            }

            if (!video.videoWidth) {

                alert(
                    "Camera is still starting. " +
                    "Please wait a moment and try again."
                );

                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const context = canvas.getContext("2d");

            context.drawImage(
                video,
                0,
                0,
                canvas.width,
                canvas.height
            );

            canvas.toBlob((blob) => {

                if (!blob) {
                    alert("Could not capture image.");
                    return;
                }

                selectedImage = new File(
                    [blob],
                    "camera_fruit.png",
                    {
                        type: "image/png"
                    }
                );

                const imageURL =
                    URL.createObjectURL(selectedImage);

                preview.innerHTML = `
                    <img
                        class="preview-image"
                        src="${imageURL}"
                        alt="Captured fruit"
                    >

                    <p class="success-text">
                        ✓ Fruit photo captured successfully
                    </p>
                `;

                stopCamera();

                resetPrediction();

            }, "image/png");

        });

    }


    // ==============================
    // CLOSE CAMERA
    // ==============================

    if (closeCamera) {

        closeCamera.addEventListener(
            "click",
            stopCamera
        );

    }


    function stopCamera() {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(track => track.stop());

            cameraStream = null;
        }

        if (video) {
            video.srcObject = null;
        }

        if (cameraArea) {
            cameraArea.style.display = "none";
        }

        if (openCamera) {
            openCamera.textContent = "📷 Open Camera";
        }

    }


    // ==============================
    // ANALYZE FRUIT
    // ==============================

    if (analyzeButton) {

        analyzeButton.addEventListener(
            "click",
            analyzeFruit
        );

    }


    async function analyzeFruit() {

        if (!selectedImage) {

            alert(
                "Please upload or capture a fruit image first."
            );

            return;
        }


        // Loading state
        analyzeButton.disabled = true;

        analyzeButton.textContent =
            "🔄 Analyzing Fruit...";


        const formData = new FormData();

        formData.append(
            "image",
            selectedImage
        );


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok || !data.success) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );

            }


            // Display AI result
            displayPrediction(data);


        }

        catch (error) {

            console.error(
                "Prediction Error:",
                error
            );

            alert(
                "AI analysis failed.\n\n" +
                error.message
            );

        }

        finally {

            analyzeButton.disabled = false;

            analyzeButton.textContent =
                "Analyze Fruit →";

        }

    }


    // ==============================
    // DISPLAY PREDICTION
    // ==============================

    function displayPrediction(data) {

        const className =
            data.class_name || "Unknown";

        const confidence =
            Number(data.confidence || 0);


        // --------------------------------
        // Freshness score
        // --------------------------------

        let score = calculateFreshnessScore(
            className,
            confidence
        );

        score = Math.round(score);


        freshnessScore.textContent = score;

        scoreText.textContent =
            `${score} / 100`;

        freshnessMeter.style.width =
            `${score}%`;

        freshnessRing.style.background =
            `conic-gradient(
                var(--mint) 0 ${score}%,
                rgba(112,225,176,.1) ${score}%
            )`;


        // --------------------------------
        // Class formatting
        // --------------------------------

        const readableClass =
            formatClassName(className);


        predictionValue.textContent =
            readableClass;


        confidenceValue.textContent =
            `${confidence.toFixed(1)}%`;


        // --------------------------------
        // Fresh / Rotten logic
        // --------------------------------

        const isFresh =
            className.toLowerCase()
                .includes("fresh");


        if (isFresh) {

            conditionValue.textContent =
                "Low";

            conditionValue.className =
                "value good";

            shelfLifeValue.textContent =
                "Approx. 3–5 days";

            resultTitle.textContent =
                "Fruit appears relatively fresh";

            resultDescription.textContent =
                "The model classified the uploaded image " +
                "as a fresh fruit category. Remaining shelf life " +
                "is an approximate estimate and depends on storage.";

            recommendation.textContent =
                "Good candidate for normal use";

            recommendationText.textContent =
                "Continue appropriate storage and monitor " +
                "visible changes over time.";

            whyPrediction.textContent =
                "Why? • fresh visual category • low visible deterioration";

        }

        else {

            conditionValue.textContent =
                "High";

            conditionValue.className =
                "value danger";

            shelfLifeValue.textContent =
                "Very limited";

            resultTitle.textContent =
                "Visible deterioration detected";

            resultDescription.textContent =
                "The model classified the uploaded image " +
                "as a deteriorated fruit category. Do not treat " +
                "this prediction as a food-safety certification.";

            recommendation.textContent =
                "Prioritize inspection";

            recommendationText.textContent =
                "Inspect the fruit carefully and follow appropriate " +
                "food-safety practices before consumption.";

            whyPrediction.textContent =
                "Why? • deteriorated visual category • high spoilage risk";

        }

    }


    // ==============================
    // FRESHNESS SCORE
    // ==============================

    function calculateFreshnessScore(
        className,
        confidence
    ) {

        const name =
            className.toLowerCase();


        if (name.includes("fresh")) {

            return Math.max(
                70,
                Math.min(
                    98,
                    confidence
                )
            );

        }


        return Math.max(
            5,
            Math.min(
                35,
                100 - confidence
            )
        );

    }


    // ==============================
    // FORMAT CLASS NAME
    // ==============================

    function formatClassName(className) {

        return className
            .replace(/_/g, " ")
            .replace(/([a-z])([A-Z])/g, "$1 $2")
            .replace(/\b\w/g, letter =>
                letter.toUpperCase()
            );

    }


    // ==============================
    // RESET RESULT
    // ==============================

    function resetPrediction() {

        freshnessScore.textContent = "82";

        scoreText.textContent =
            "82 / 100";

        freshnessMeter.style.width =
            "82%";

        freshnessRing.style.background =
            `conic-gradient(
                var(--mint) 0 82%,
                rgba(112,225,176,.1) 82%
            )`;

        predictionValue.textContent =
            "Waiting";

        conditionValue.textContent =
            "Waiting";

        shelfLifeValue.textContent =
            "—";

        confidenceValue.textContent =
            "—";

        resultTitle.textContent =
            "Ready for analysis";

        resultDescription.textContent =
            "Upload or capture a fruit image and click " +
            "Analyze Fruit to run the AI model.";

        recommendation.textContent =
            "Waiting for analysis";

        recommendationText.textContent =
            "The recommendation will appear after the " +
            "AI analysis is completed.";

        whyPrediction.textContent =
            "Why? • awaiting image analysis";

    }


    // ==============================
    // TABS
    // ==============================

    document
        .querySelectorAll(".tab")
        .forEach(tab => {

            const activate = () => {

                document
                    .querySelectorAll(".tab")
                    .forEach(item =>
                        item.classList.remove("active")
                    );

                tab.classList.add("active");

            };


            tab.addEventListener(
                "click",
                activate
            );


            tab.addEventListener(
                "keydown",
                event => {

                    if (
                        event.key === "Enter" ||
                        event.key === " "
                    ) {

                        event.preventDefault();

                        activate();

                    }

                }
            );

        });

});