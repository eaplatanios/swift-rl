#if GLFW
import CRetro
import Foundation
import ReinforcementLearning
import TensorFlow

public class SingleImageRenderer: Renderer {
  public typealias Data = ShapedArray<UInt8>

  public let framesPerSecond: Double?

  private var window: OpaquePointer?
  private var frameBuffer: GLuint = 0
  private var texture: GLuint = 0

  private let initialMaxWidth: Int32

  public private(set) var isOpen: Bool = true

  public init(initialMaxWidth: Int32, framesPerSecond: Double? = nil) throws {
    self.initialMaxWidth = initialMaxWidth
    self.framesPerSecond = framesPerSecond
  }

  deinit {
    closeWindow()
  }

  public func render(_ data: ShapedArray<UInt8>) throws {
    if !isOpen { return }

    if let fps = framesPerSecond {
      Thread.sleep(forTimeInterval: 1 / fps)
    }

    if self.window == nil {
      var width = Int32(data.shape[1])
      var height = Int32(data.shape[0])
      if width > initialMaxWidth {
        let scale = Float(initialMaxWidth) / Float(width)
        width = Int32(scale * Float(width))
        height = Int32(scale * Float(height))
      }

      try createWindow(width: width, height: height)
    }

    if let window = self.window {
      if glfwWindowShouldClose(window) > 0 {
        closeWindow()
        exit(0)
      }

      // var texture: GLuint = 0
      glfwMakeContextCurrent(window)
      glClear(UInt32(GL_COLOR_BUFFER_BIT))
      
      // Generate the image texture.
      try SingleImageRenderer.preprocessData(data).withUnsafeBufferPointer {
        glTexImage2D(
          GLenum(GL_TEXTURE_2D), 0, GL_RGB8, GLsizei(data.shape[1]), GLsizei(data.shape[0]), 
          0, GLenum(GL_RGB), GLenum(GL_UNSIGNED_BYTE), $0.baseAddress)
      }

      // Resize and render the texture.
      var width: GLsizei = 0
      var height: GLsizei = 0
      glfwGetFramebufferSize(window, &width, &height)
      glBlitFramebuffer(
        0, 0, GLsizei(data.shape[1]), GLsizei(data.shape[0]), 0, 0, 
        width, height, GLenum(GL_COLOR_BUFFER_BIT), GLenum(GL_LINEAR))
      
      // Swap the OpenGL front and back buffers to show the image.
      glfwSwapBuffers(window)
      glfwPollEvents()
    }
  }

  public func createWindow(width: Int32, height: Int32) throws {
    // Initialize GLFW.
    if glfwInit() == 0 {
      throw RetroError.GLFWError("Failed to initialize GLFW.")
    }

    // Open a new window.
    guard let window = glfwCreateWindow(width, height, "Gym Retro", nil, nil) else {
      glfwTerminate()
      throw RetroError.GLFWError("Failed to open a GLFW window.")
    }

    self.window = window

    glfwMakeContextCurrent(window)
    
    // Generate a frame buffer.
    glGenFramebuffers(1, &frameBuffer)
    glBindFramebuffer(GLenum(GL_READ_FRAMEBUFFER), frameBuffer)

    // Generate a texture.
    glGenTextures(1, &texture)
    glBindTexture(GLenum(GL_TEXTURE_2D), texture)
    glTexParameteri(GLenum(GL_TEXTURE_2D), GLenum(GL_TEXTURE_MAG_FILTER), GL_NEAREST)
    glTexParameteri(GLenum(GL_TEXTURE_2D), GLenum(GL_TEXTURE_MIN_FILTER), GL_NEAREST)
    
    // Bind the texture to the frame buffer.
    glFramebufferTexture2D(
      GLenum(GL_READ_FRAMEBUFFER), GLenum(GL_COLOR_ATTACHMENT0), 
      GLenum(GL_TEXTURE_2D), texture, 0)
  }

  public func closeWindow() {
    if let window = self.window {
      glDeleteTextures(1, &texture)
      glDeleteFramebuffers(1, &frameBuffer)
      glfwDestroyWindow(window)
      glfwPollEvents()
      glfwTerminate()
      self.isOpen = false
      self.window = nil
    }
  }

  private static func preprocessData(_ data: ShapedArray<UInt8>) throws -> [UInt8] {
    precondition(data.rank == 3 && data.shape[2] == 3, "Data must have shape '[Height, Width, 3]'.")
    let rowSize = data.shape[1] * data.shape[2]
    let scalars = data.scalars
    var preprocessed = [UInt8]()
    preprocessed.reserveCapacity(scalars.count)
    for row in 1...data.shape[0] {
      let index = (data.shape[0] - row) * rowSize
      preprocessed.append(contentsOf: scalars[index..<(index + rowSize)])
    }
    return preprocessed
  }
}

#endif
