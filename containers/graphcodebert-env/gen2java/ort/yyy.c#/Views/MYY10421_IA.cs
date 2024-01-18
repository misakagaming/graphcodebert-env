// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: MYY10421_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:31
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: MYY10421_IA
  /// </summary>
  [Serializable]
  public class MYY10421_IA : ViewBase, IImportView
  {
    private static MYY10421_IA[] freeArray = new MYY10421_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP
    //        Type: CANAM_XML
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpCanamXmlXmlBuffer
    /// </summary>
    private char _ImpCanamXmlXmlBuffer_AS;
    /// <summary>
    /// Attribute missing flag for: ImpCanamXmlXmlBuffer
    /// </summary>
    public char ImpCanamXmlXmlBuffer_AS {
      get {
        return(_ImpCanamXmlXmlBuffer_AS);
      }
      set {
        _ImpCanamXmlXmlBuffer_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpCanamXmlXmlBuffer
    /// Domain: Text
    /// Length: 4094
    /// Varying Length: Y
    /// </summary>
    private string _ImpCanamXmlXmlBuffer;
    /// <summary>
    /// Attribute for: ImpCanamXmlXmlBuffer
    /// </summary>
    public string ImpCanamXmlXmlBuffer {
      get {
        return(_ImpCanamXmlXmlBuffer);
      }
      set {
        _ImpCanamXmlXmlBuffer = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public MYY10421_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public MYY10421_IA( MYY10421_IA orig )
    {
      ImpCanamXmlXmlBuffer_AS = orig.ImpCanamXmlXmlBuffer_AS;
      ImpCanamXmlXmlBuffer = orig.ImpCanamXmlXmlBuffer;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static MYY10421_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new MYY10421_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new MYY10421_IA());
          }
          else 
          {
            MYY10421_IA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new MYY10421_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpCanamXmlXmlBuffer_AS = ' ';
      ImpCanamXmlXmlBuffer = "";
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((MYY10421_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( MYY10421_IA orig )
    {
      ImpCanamXmlXmlBuffer_AS = orig.ImpCanamXmlXmlBuffer_AS;
      ImpCanamXmlXmlBuffer = orig.ImpCanamXmlXmlBuffer;
    }
  }
}
