package com.mzm.sample.esrgan

import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import com.mzm.sample.esrgan.ml.Esrgan
import org.tensorflow.lite.support.image.TensorImage

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val downsampled = BitmapFactory.decodeResource(resources, R.drawable.downsampled)
        val imageViewOriginal: ImageView = findViewById(R.id.imageview_original)
        val imageViewEnhanced: ImageView = findViewById(R.id.imageview_enhanced)
        imageViewOriginal.setImageBitmap(downsampled)

        val model = Esrgan.newInstance(this)

// Creates inputs for reference.
        val originalImage = TensorImage.fromBitmap(downsampled)

// Runs model inference and gets result.
        val outputs = model.process(originalImage)
        val enhancedImage = outputs.enhancedImageAsTensorImage
        val enhancedImageBitmap = enhancedImage.bitmap

// Releases model resources if no longer used.
        model.close()
        imageViewEnhanced.setImageBitmap(enhancedImageBitmap)

    }
}