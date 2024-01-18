// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: EYYY0311_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:19
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
// START OF EXPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: EYYY0311_OA
  /// </summary>
  [Serializable]
  public class EYYY0311_OA : ViewBase, IExportView
  {
    private static EYYY0311_OA[] freeArray = new EYYY0311_OA[30];
    private static int countFree = 0;
    
    // Entity View: EXP
    //        Type: TYPE
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpTypeTreferenceId
    /// </summary>
    private char _ExpTypeTreferenceId_AS;
    /// <summary>
    /// Attribute missing flag for: ExpTypeTreferenceId
    /// </summary>
    public char ExpTypeTreferenceId_AS {
      get {
        return(_ExpTypeTreferenceId_AS);
      }
      set {
        _ExpTypeTreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpTypeTreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ExpTypeTreferenceId;
    /// <summary>
    /// Attribute for: ExpTypeTreferenceId
    /// </summary>
    public string ExpTypeTreferenceId {
      get {
        return(_ExpTypeTreferenceId);
      }
      set {
        _ExpTypeTreferenceId = value;
      }
    }
    // Entity View: EXP_ERROR
    //        Type: DONT_CHANGE_TEXT
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorDontChangeTextText2
    /// </summary>
    private char _ExpErrorDontChangeTextText2_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorDontChangeTextText2
    /// </summary>
    public char ExpErrorDontChangeTextText2_AS {
      get {
        return(_ExpErrorDontChangeTextText2_AS);
      }
      set {
        _ExpErrorDontChangeTextText2_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorDontChangeTextText2
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorDontChangeTextText2;
    /// <summary>
    /// Attribute for: ExpErrorDontChangeTextText2
    /// </summary>
    public string ExpErrorDontChangeTextText2 {
      get {
        return(_ExpErrorDontChangeTextText2);
      }
      set {
        _ExpErrorDontChangeTextText2 = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public EYYY0311_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public EYYY0311_OA( EYYY0311_OA orig )
    {
      ExpTypeTreferenceId_AS = orig.ExpTypeTreferenceId_AS;
      ExpTypeTreferenceId = orig.ExpTypeTreferenceId;
      ExpErrorDontChangeTextText2_AS = orig.ExpErrorDontChangeTextText2_AS;
      ExpErrorDontChangeTextText2 = orig.ExpErrorDontChangeTextText2;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static EYYY0311_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new EYYY0311_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new EYYY0311_OA());
          }
          else 
          {
            EYYY0311_OA result = freeArray[--countFree];
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
      return(new EYYY0311_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpTypeTreferenceId_AS = ' ';
      ExpTypeTreferenceId = "00000000000000000000";
      ExpErrorDontChangeTextText2_AS = ' ';
      ExpErrorDontChangeTextText2 = "  ";
    }
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IExportView orig )
    {
      this.CopyFrom((EYYY0311_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( EYYY0311_OA orig )
    {
      ExpTypeTreferenceId_AS = orig.ExpTypeTreferenceId_AS;
      ExpTypeTreferenceId = orig.ExpTypeTreferenceId;
      ExpErrorDontChangeTextText2_AS = orig.ExpErrorDontChangeTextText2_AS;
      ExpErrorDontChangeTextText2 = orig.ExpErrorDontChangeTextText2;
    }
  }
}
