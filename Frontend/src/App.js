import React, { useState } from "react";
import './App.css';

function WardrobeFeller() {
  const [inputImage, setInputImage] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [gender, setGender] = useState("");
  const [size, setSize] = useState("");
  const [category, setDressCategory] = useState("");  // Changed variable name
  const [isLoading, setIsLoading] = useState(false);

  // Handle input image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setInputImage(imageUrl);
    }
  };

  const handleSubmit = async () => {
    if (!inputImage || !gender || !size || !category) {
      alert("Please fill in all the fields and upload an image.");
      return;
    }
  
    setIsLoading(true);
  
    const formData = new FormData();
    formData.append("cloth_image", document.querySelector('input[type="file"]').files[0]);
    formData.append("gender", gender);
    formData.append("size", size);
    formData.append("category", category);
    console.log("API URL:", process.env.REACT_APP_API_URL);

  
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      
      if (response.ok) {
        const timestamp = new Date().getTime(); 
        setOutputImage(`result_image.jpg?_=${timestamp}`);
      } else {
        alert(data.error || "An error occurred while processing the image.");
      }
    } catch (error) {
      alert("An error occurred: " + error.message);
    }
  
    setIsLoading(false);
  };
  

  return (
    <div style={styles.mainContainer}>
      <h1 style={styles.title}>Wardrobe Feeler</h1>

      <div style={styles.contentContainer}>
      <div style={styles.inputSection}>
  <h3 style={styles.outputHeader}>Upload the dress Image:</h3>
  <div style={styles.box}>
    <p>Upload or browse from computer</p>
    <input
      type="file"
      accept="image/*"
      name="cloth_image"
      onChange={handleImageUpload}
      style={styles.uploadInput}
    />
  </div>
  {inputImage && (
    <div style={styles.previewBox}>
      <img
        src={inputImage}
        alt="Selected Input"
        style={styles.inputImagePreview}
      />
    </div>
  )}

          <div style={styles.field}>
            <label>Gender:</label>
            <select
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              style={styles.select}
            >
              <option value="">Select...</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Kid">Kid</option>
            </select>
          </div>

          <div style={styles.field}>
            <label>Size:</label>
            <div style={styles.sizeContainer}>
              {["S", "M", "L"].map((s) => (
                <button
                  key={s}
                  onClick={() => setSize(s)}
                  style={{
                    ...styles.sizeButton,
                    backgroundColor: size === s ? "#aaa" : "#fff",
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div style={styles.field}>
            <label>Dress Category:</label> 
            <select
              value={category}  
              onChange={(e) => setDressCategory(e.target.value)}  
              style={styles.select}
            >
              <option value="">Select...</option>
              <option value="Upper-Body">Upper-Body</option>
              <option value="Lower-Body">Lower-Body</option>
              <option value="Dress">Dress</option>
            </select>
          </div>

          <button onClick={handleSubmit} style={styles.saveButton} disabled={isLoading}>
            {isLoading ? "Uploading..." : "SUBMIT"}
          </button>
        </div>

        <div style={styles.outputSection}>
          <h3 style={styles.outputHeader}>Output Image:</h3>
          <div style={styles.outputBox}>
          {outputImage ? (
  <img
    src={`http://localhost:5000/output?path=${outputImage}`}
    alt="Output"
    style={styles.outputImage}
    onError={() => console.error("Image failed to load:", outputImage)}
  />
) : (
  <p>No output image available</p>
)}

          </div>
          <button style={styles.saveButton}>SAVE</button>
        </div>
      </div>
    </div>
  );
}

const styles = {
 mainContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    fontFamily: "Arial, sans-serif",
    padding: "20px",
    margin: "0 auto 20px",
    minHeight: "100vh", 
    backgroundImage: "url('bg_img.png')", 
    backgroundSize: "cover",
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
  },
  title: {
    backgroundColor: "black",
    color: "white",
    textAlign: "center",
    padding: "15px 20px",
    borderRadius: "8px",
    width: "100%",
    fontSize: "2.5rem",
    fontWeight: "bold",
    textShadow: "2px 2px 4px rgba(0, 0, 0, 0.5)",
    marginBottom: "20px",
  },
  contentContainer: {
    display: "flex",
    justifyContent: "space-between",
    width: "90%",
    marginTop: "20px",
  },
  inputSection: {
    display: "flex",
    flexDirection: "column",
    gap: "15px",
    alignItems: "flex-start",
    width: "45%", // Left section width
  },
  previewBox: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "10px",
    marginTop: "10px",
    width: "150px",   // Set the box width for small preview
    height: "150px",  // Set the box height for small preview
    overflow: "hidden",
    boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.2)",
    borderRadius: "5px",
    backgroundColor: "transparent",
  },
  
  inputImagePreview: {
    maxWidth: "100%",
    maxHeight: "100%",
    objectFit: "contain", // Maintain aspect ratio and fit inside the box
    borderRadius: "5px",
  },
  box: {
    border: "1px solid black",
    padding: "10px",
    width: "100%",
    textAlign: "center",
    boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.2)",
  },
  uploadInput: {
    marginTop: "10px",
  },
  field: {
    display: "flex",
    flexDirection: "column",
    gap: "5px",
  },
  select: {
    width: "100px",
    padding: "5px",
  },
  sizeContainer: {
    display: "flex",
    gap: "10px",
  },
  sizeButton: {
    padding: "5px 10px",
    cursor: "pointer",
    border: "1px solid #000",
    borderRadius: "5px",
  },
  outputSection: {
    width: "45%", // Right section width
    textAlign: "center",
  },
  outputHeader: {
    fontWeight: "bold",
  },
  outputBox: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    border: "1px solid black",
    padding: "10px",
    marginBottom: "10px",
    boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.2)",
    width: "auto", 
    height: "auto", 
    overflow: "hidden", 
  },
  
  outputImage: {
    maxWidth: "100%", 
    maxHeight: "100%", 
    objectFit: "contain", 
  },
  saveButton: {
    padding: "10px 20px",
    backgroundColor: "black",
    color: "white",
    border: "none",
    cursor: "pointer",
    fontWeight: "bold",
    borderRadius: "5px",
  },
};


export default WardrobeFeller;
