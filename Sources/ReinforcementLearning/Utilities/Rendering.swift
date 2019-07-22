// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import GLFW
import Foundation
import TensorFlow

public class ImageRenderer {
  public let framesPerSecond: Double?

  @usableFromInline internal let initialMaxWidth: Int32
  @usableFromInline internal var window: OpaquePointer?
  @usableFromInline internal var frameBuffer: GLuint = 0
  @usableFromInline internal var texture: GLuint = 0
  @usableFromInline internal var isOpen: Bool = true

  @inlinable
  public init(initialMaxWidth: Int32 = 800, framesPerSecond: Double? = nil) {
    self.initialMaxWidth = initialMaxWidth
    self.framesPerSecond = framesPerSecond
  }

  @inlinable
  deinit {
    closeWindow()
  }

  @inlinable
  public func render(_ data: ShapedArray<UInt8>) throws {
    if !isOpen { return }
    if let fps = framesPerSecond { Thread.sleep(forTimeInterval: 1 / fps) }

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
      if glfwWindowShouldClose(window) > 0 { closeWindow(); exit(0) }
      glfwMakeContextCurrent(window)
      glClear(UInt32(GL_COLOR_BUFFER_BIT))

      // Generate the image texture.
      try ImageRenderer.preprocessData(data).withUnsafeBufferPointer {
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

  @inlinable
  public func createWindow(width: Int32, height: Int32) throws {
    // Initialize GLFW.
    if glfwInit() == 0 {
      throw RLError.GLFWError("Failed to initialize GLFW.")
    }

    // Open a new window.
    guard let window = glfwCreateWindow(width, height, "Gym Retro", nil, nil) else {
      glfwTerminate()
      throw RLError.GLFWError("Failed to open a GLFW window.")
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

  @inlinable
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

  @inlinable
  internal static func preprocessData(_ data: ShapedArray<UInt8>) throws -> [UInt8] {
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

public protocol GLFWScene {
  func draw()
}

extension GLFWScene {
  @inlinable
  public func render(in window: GLFWWindow) {
    if !window.isOpen { return }
    if let fps = window.framesPerSecond { Thread.sleep(forTimeInterval: 1 / fps) }
    if let w = window.window {
      // TODO: Should the following be exiting the running problem?
      if glfwWindowShouldClose(w) > 0 { window.close(); exit(0) }
      glfwMakeContextCurrent(w)
      glClearColor(1, 1, 1, 1)
      glClear(GLbitfield(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT))
      draw()
      glfwSwapBuffers(w)
      glfwPollEvents()
    }
  }
}

public class GLFWWindow {
  public let name: String

  @usableFromInline internal let width: Int
  @usableFromInline internal let height: Int
  @usableFromInline internal let framesPerSecond: Double?
  @usableFromInline internal var window: OpaquePointer?
  @usableFromInline internal var isOpen: Bool = true

  @inlinable
  public init(name: String, width: Int, height: Int, framesPerSecond: Double? = nil) throws {
    self.name = name
    self.width = width
    self.height = height
    self.framesPerSecond = framesPerSecond

    // Initialize GLFW.
    if glfwInit() == 0 { throw RLError.GLFWError("Failed to initialize GLFW.") }

    // Open a new window.
    guard let window = glfwCreateWindow(Int32(width), Int32(height), name, nil, nil) else {
      glfwTerminate()
      throw RLError.GLFWError("Failed to open a GLFW window.")
    }

    self.window = window
    glfwMakeContextCurrent(window)
    glEnable(GLenum(GL_BLEND))
    glBlendFunc(GLenum(GL_SRC_ALPHA), GLenum(GL_ONE_MINUS_SRC_ALPHA))
    glViewport(0, 0, Int32(width), Int32(height))
    glMatrixMode(GLenum(GL_PROJECTION))
    glLoadIdentity()
    glOrtho(0, GLdouble(width), 0, GLdouble(height), 0, 1)
  }

  @inlinable
  deinit {
    close()
  }

  @inlinable
  public func close() {
    if let window = self.window {
      glfwDestroyWindow(window)
      glfwPollEvents()
      glfwTerminate()
      self.isOpen = false
      self.window = nil
    }
  }
}

public protocol GLFWAttribute: AnyObject {
  func enable()
  func disable()
}

public class GLFWTransform: GLFWAttribute {
  public var translation: (Float, Float)
  public var rotation: Float
  public var scale: (Float, Float)

  @inlinable
  public init(
    translation: (Float, Float) = (0.0, 0.0),
    rotation: Float = 0.0,
    scale: (Float, Float) = (1.0, 1.0)
  ) {
    self.translation = translation
    self.rotation = rotation
    self.scale = scale
  }

  @inlinable
  public func enable() {
    glPushMatrix()
    glTranslatef(translation.0, translation.1, 0)
    glRotatef(rotation * 180 / Float.pi, 0, 0, 1)
    glScalef(scale.0, scale.1, 1)
  }

  @inlinable
  public func disable() {
    glPopMatrix()
  }
}

public class GLFWColor: GLFWAttribute {
  public let red: Float
  public let green: Float
  public let blue: Float

  @inlinable
  public init(red: Float, green: Float, blue: Float) {
    self.red = red
    self.green = green
    self.blue = blue
  }

  @inlinable
  public func enable() {
    glColor3f(red, green, blue)
  }

  @inlinable
  public func disable() { }
}

public class GLFWLineStyle: GLFWAttribute {
  public let pattern: UInt16

  @inlinable
  public init(_ pattern: UInt16) {
    self.pattern = pattern
  }

  @inlinable
  public func enable() {
    glEnable(GLenum(GL_LINE_STIPPLE))
    glLineStipple(1, pattern)
  }

  @inlinable
  public func disable() {
    glDisable(GLenum(GL_LINE_STIPPLE))
  }
}

public class GLFWLineWidth: GLFWAttribute {
  public let width: Float

  @inlinable
  public init(_ width: Float) {
    self.width = width
  }

  @inlinable
  public func enable() {
    glLineWidth(width)
  }

  @inlinable
  public func disable() { }
}

public protocol GLFWGeometry: AnyObject {
  var attributes: [GLFWAttribute] { get set }

  func render()
}

extension GLFWGeometry {
  @inlinable
  public func renderWithAttributes() {
    attributes.reversed().forEach { $0.enable() }
    render()
    attributes.forEach { $0.disable() }
  }
}

public class GLFWCompoundGeometry: GLFWGeometry {
  public var attributes: [GLFWAttribute] = []
  public var components: [GLFWGeometry]

  @inlinable
  public init(_ components: GLFWGeometry...) {
    self.components = components
    attributes = components.flatMap { $0.attributes }
  }

  @inlinable
  public func render() {
    components.forEach { $0.render() }
  }

  @inlinable
  public func renderWithAttributes() {
    components.forEach { $0.renderWithAttributes() }
  }
}

public class GLFWPoint: GLFWGeometry {
  public var attributes: [GLFWAttribute] = []
  public var coordinates: (Float, Float, Float)

  @inlinable
  public init(coordinates: (Float, Float, Float) = (0, 0, 0)) {
    self.coordinates = coordinates
  }

  @inlinable
  public func render() {
    glBegin(GLenum(GL_POINTS))
    glVertex3f(0, 0, 0)
    glEnd()
  }
}

public class GLFWLine: GLFWGeometry {
  public var attributes: [GLFWAttribute] = []
  public var start: (Float, Float)
  public var end: (Float, Float)

  @inlinable
  public init(start: (Float, Float), end: (Float, Float)) {
    self.start = start
    self.end = end
  }

  @inlinable
  public func render() {
    glBegin(GLenum(GL_LINES))
    glVertex2f(start.0, start.1)
    glVertex2f(end.0, end.1)
    glEnd()
  }
}

public class GLFWPolygon: GLFWGeometry {
  public var attributes: [GLFWAttribute] = []
  public var vertices: [(Float, Float)]

  @inlinable
  public init(vertices: [(Float, Float)]) {
    self.vertices = vertices
  }

  @inlinable
  public func render() {
    switch vertices.count {
    case 4: glBegin(GLenum(GL_QUADS))
    case 4...: glBegin(GLenum(GL_POLYGON))
    case _: glBegin(GLenum(GL_TRIANGLES))
    }
    vertices.forEach { glVertex3f($0.0, $0.1, 0) }
    glEnd()
  }
}

public class GLFWPolyLine: GLFWGeometry {
  public var attributes: [GLFWAttribute] = []
  public var vertices: [(Float, Float)]
  public var closed: Bool

  @inlinable
  public init(vertices: [(Float, Float)], closed: Bool) {
    self.vertices = vertices
    self.closed = closed
  }

  @inlinable
  public func render() {
    glBegin(GLenum(closed ? GL_LINE_LOOP : GL_LINE_STRIP))
    vertices.forEach { glVertex3f($0.0, $0.1, 0) }
    glEnd()
  }
}

public class GLFWCircle: GLFWGeometry {
  public var attributes: [GLFWAttribute] = []
  public var radius: Float = 10
  public var resolution: Int = 30
  public var filled: Bool = true

  @inlinable
  public func render() {
    let vertices = (0..<resolution).map { i -> (Float, Float) in
      let angle = 2 * Float.pi * Float(i) / Float(resolution)
      return (cos(angle) * radius, sin(angle) * radius)
    }
    if filled {
      GLFWPolygon(vertices: vertices).render()
    } else {
      GLFWPolyLine(vertices: vertices, closed: true).render()
    }
  }
}

// public class GLFWImage: GLFWGeometry {
//   public var attributes: [GLFWAttribute] = []
//   public var image: ShapedArray<UInt8>

//   public init(_ image: ShapedArray<UInt8>) {
//     precondition(
//       image.rank == 3 && image.shape[2] == 3,
//       "The image array must have shape '[Height, Width, 3]'.")
//     self.image = image
//   }

//   public func render(in window: OpaquePointer?) {
//     // Generate the image texture.    
//     let rowSize = image.shape[1] * image.shape[2]
//     let scalars = image.scalars
//     var preprocessed = [UInt8]()
//     preprocessed.reserveCapacity(scalars.count)
//     for row in 1...image.shape[0] {
//       let index = (image.shape[0] - row) * rowSize
//       preprocessed.append(contentsOf: scalars[index..<(index + rowSize)])
//     }
//     preprocessed.withUnsafeBufferPointer {
//       glTexImage2D(
//         GLenum(GL_TEXTURE_2D), 0, GL_RGB8, GLsizei(image.shape[1]), GLsizei(image.shape[0]),
//         0, GLenum(GL_RGB), GLenum(GL_UNSIGNED_BYTE), $0.baseAddress)
//     }

//     // Resize and render the texture.
//     var width: GLsizei = 0
//     var height: GLsizei = 0
//     glfwGetFramebufferSize(window, &width, &height)
//     glBlitFramebuffer(
//       0, 0, GLsizei(image.shape[1]), GLsizei(image.shape[0]), 0, 0,
//       width, height, GLenum(GL_COLOR_BUFFER_BIT), GLenum(GL_LINEAR))
//   }
// }
